# SPEC_01 summary

- N = 783, D = 24 (m3 v9 783-rxn bundle)
- cond(X.T@X) = 2.169e+16, rank = 25
- alpha=0 admissible: False

## Selected alpha per channel/barrier

| channel | a*_CV | a*_GCV | NMAE@a=1 | NMAE@a* | delta (a=1 -> a*) |
|---|---|---|---|---|---|
| strain | 4.64e+00 | 6.81e-01 | 0.497 | 0.497 | +0.001 |
| Pauli | 4.64e-02 | 3.16e-02 | 0.462 | 0.458 | +0.004 |
| elst | 2.15e-02 | 3.16e-02 | 0.484 | 0.479 | +0.004 |
| oi | 1.47e-01 | 4.64e-02 | 0.394 | 0.394 | +0.000 |
| disp | 6.81e-06 | 2.15e-01 | 0.229 | 0.228 | +0.000 |
| barrier | 1.00e-06 | 6.81e-01 | 0.366 | 0.366 | +0.000 |

## Scope note (delta interaction)

`b` is only the ridge baseline; the residual `delta = y - b` is what MLP head learns.
The alpha that minimizes b-alone NMAE is NOT necessarily optimal for the b+delta system.
System-level alpha selection belongs to SPEC_02.