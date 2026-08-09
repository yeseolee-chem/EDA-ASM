# CAVEATS — things that must be stated when this data is published

Measured against the actual ORCA 6.1.1 outputs in `data/rxn_*/eda.out`, not assumed.

---

## 1. The dispersion channel is very nearly an identity, not a physical finding

`Delta Dispersion` is

    D3(BJ)[complex] − D3(BJ)[frag1] − D3(BJ)[frag2]

D3(BJ) is an **empirical, force-field-like correction evaluated analytically from
the nuclear coordinates alone**. No wavefunction enters it. ORCA reports it in a
separate `$VdW_Correction` block in `eda.property.txt`, confirming it is computed
outside the SCF.

Consequence: a model that predicts this channel accurately has learned a closed-form
geometric function, not quantum chemistry. Reporting "disp is the easiest channel"
as a physical result is not defensible — a reviewer will say the label is analytic.

**Recommended handling.** Compute a D3(BJ)-only oracle baseline (e.g. `simple-dftd3`
on the same geometries and fragment partition) and show its error is ~0. Then label
`disp` explicitly as a *sanity channel* in the headline table rather than a learning
target. Any residual error on `disp` measures model capacity, not physics.

This strengthens rather than weakens the claim: it separates the channels that are
genuinely a learning problem from the one that is not, and it explains both ends of
the channel-difficulty ordering.

---

## 2. No BSSE / counterpoise correction

Verified: no `counterpoise`, `bsse`, or ghost-atom directives appear in any
`eda.inp` or in the outputs. With B3LYP/def2-TZVP a basis-set superposition error
of roughly 1–2 kcal/mol remains in ΔE_int.

For ML labels this is largely a smooth offset and mostly harmless, but it does **not**
load equally on the individual channels — it enters mainly through the orbital and
electrostatic terms. State it in the paper. Do not claim counterpoise-quality
interaction energies.

---

## 3. Most SCFs converge on the energy criterion, not on all TightSCF criteria

Measured over the first 116 completed reactions: **83 (72%)** end with

    **** Energy Check signals convergence ****

which means the SCF stopped once the energy criterion was met while the
density / DIIS / orbital-gradient criteria were still above the TightSCF thresholds.

The total energy is second order in the density error, so `Bond Energy` is safe. The
ETS channels are built from the density and Fock matrices directly and are **first
order**, and this system cancels hard — Pauli of about +237 kcal/mol against the
attractive terms leaves only −19.5. A density error that is negligible for the sum
can be significant for the parts.

**MEASURED — the concern does not materialise.** 20 reactions spanning the
basis-function range were re-run with `VeryTightSCF` (`scripts/validate_scf.pbs`) and
differenced channel by channel against their `TightSCF` counterparts on the Hartree
column (`scripts/compare_scf.py`). 15 of the 20 had exited on the energy check.

    channel   max|drift|   mean|drift|   (kcal/mol)
    bond        0.0178       0.0053
    elst        0.0172       0.0053
    pauli       0.0043       0.0009
    xc          0.0026       0.0005
    orb         0.0017       0.0004
    cpcm        0.0007       0.0002
    disp        0.0000       0.0000
    smd         0.0000       0.0000

Largest drift anywhere: **0.018 kcal/mol**, about a third of the 0.05 pass criterion.
Pauli — the term feared most because of the +237 / −19.5 cancellation — is among the
most stable at 0.004. TightSCF is sufficient; production does not need re-running.

`disp` and `smd` are identically 0.0000, which is itself the cleanest evidence for
caveat 1: D3(BJ) is analytic in the coordinates and does not respond to SCF tightening
at all.

---

## 4. Fragment SCF convergence was invisible in the first 116 reactions

ORCA runs three SCFs (complex, FRAG1, FRAG2) but `eda.out` contains only **one**
`SCF CONVERGED` block — the complex. The fragments write `eda_frag1.out` /
`eda_frag2.out`, and the main log shows only

    SPC Fragment 1                     .... done (   1424.461 sec)

The production script originally deleted the scratch directory, so a fragment SCF
that failed to converge would have passed silently — and the resulting channels would
still satisfy `sum(7 channels) == Bond Energy`, because that identity holds whether or
not the densities are converged. The sum check does **not** catch this.

Fixed: `run_one()` now reads both fragment outputs before cleanup, records the result
in `logs/frag_scf.tsv`, marks the reaction FAIL if either fragment did not converge,
and preserves the offending `eda_frag*.out` as evidence.

**Resolved.** The 119 reactions that predate the gate were quarantined to
`data_unverified/` and recomputed under the gated script.

Measured over the first 990 gated reactions: **zero** fragment SCF failures
(`logs/frag_scf.tsv`), fragment SCFs converging in 12–19 cycles. So the failure mode
being guarded against has not occurred — but it is now observable, which it was not
before.

---

## 5. The channel-sum identity holds to ~0.003 kcal/mol, not exactly

Two separate effects, measured over 982 completed reactions:

**Print rounding.** ORCA rounds the kcal/mol column to 0.01, so summing seven of
them accumulates up to 7 x 0.005 = 0.035 kcal/mol of pure print error. Checking the
identity on that column therefore cannot resolve anything finer than ~0.035.

**A real numerical residual.** Repeating the check on the Hartree column (ten
significant figures) removes the rounding entirely, and a residual remains:

    median  2.67e-03 kcal/mol
    mean    4.25e-03
    > 0.01   80 / 982   (8%)
    > 0.02   ~20 / 982  (2%)
    > 0.05    1 / 982   (rxn_3507, -0.111)

This is consistent with COSX grid and SCF convergence noise between the complex and
the two fragment calculations. It is small enough to ignore for ML labels, but it is
**not** zero and should not be described as an exact identity.

`aggregate_nodeps.py` therefore gates on the **Hartree-derived** residual
(`sum_minus_bond_eh_kcal`), reporting counts at both 0.02 and 0.05. The kcal-derived
column is kept for reference only.

**Note on an earlier claim in this file's history:** the kcal-column deviations were
initially attributed entirely to rounding, and a 0.02 threshold was dismissed as
over-strict on that basis. That was half right. Rounding does make 0.02 unusable *on
the kcal column*, but the correct response is to use the Hartree column, where 0.02
is a reasonable outlier gate rather than a false-positive generator.

## 6. Charges are (0,1)+(0,1) — correct for this subset only

Verified across all 3504 inputs: every one is `* xyz 0 1` with
`FRAG1_C 0 / FRAG2_C 0 / FRAG1_M 1 / FRAG2_M 1`, and the elements present are only
H, C, O, N, F, Cl, Br — no metals, no counterions. Fragment charges sum to the total
charge, so the partition is self-consistent.

This is the **`tt` = three_two_cycloaddition** subset: 1,3-dipole plus dipolarophile,
both neutral closed-shell. It is correct here.

It would be **wrong** for E2 / SN2, where the nucleophile or base is anionic. If this
pipeline is extended to those subsets, the fragment charges and multiplicities must be
derived per reaction, not copied from this template.

---

## 7. Protocol — confirm against any other cohort before merging

These inputs are B3LYP-D3(BJ)/def2-TZVP with CPCM(SMD, water), which `CLAUDE.md`
states matches the Espley 2024 DFT single-point protocol.

If another cohort in this project was computed at a different level (a different
functional or solvent model), the two sets **cannot be merged** — the channel values
are not comparable across functionals. Resolve this before the campaign finishes, not
after.

---

## 8. Reproducibility: the pipeline is deterministic across nodes

The 119 quarantined reactions were recomputed from the same inputs on different
compute nodes at a different time. Comparing `Bond Energy` on the Hartree column over
the 118 pairs available:

    median difference   0.00000 kcal/mol   (i.e. bit-identical for most)
    max difference      0.00114 kcal/mol
    pairs differing by more than 0.01   0 / 118

Same input, same result, regardless of which of the ~3500 KNL nodes ran it. This is
worth stating explicitly — it means the ~0.003 kcal/mol residual in caveat 5 is
systematic to the method, not run-to-run noise.
