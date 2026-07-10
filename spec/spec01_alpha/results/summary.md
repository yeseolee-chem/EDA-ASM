# SPEC_01 summary

- N = 776, D = 24 (m3 v7 776-rxn bundle)
- cond(X.T@X) = 1.597e+16, rank = 25
- alpha=0 admissible: False

## Selected alpha per channel/barrier

| channel | a*_CV | a*_GCV | NMAE@a=1 | NMAE@a* | delta (a=1 -> a*) |
|---|---|---|---|---|---|
| strain | 6.81e+00 | 3.16e-01 | 0.801 | 0.791 | +0.010 |
| Pauli | 4.64e+00 | 6.81e-01 | 0.787 | 0.777 | +0.010 |
| elst | 6.81e+00 | 6.81e-01 | 0.766 | 0.758 | +0.009 |
| oi | 3.16e+00 | 1.00e+00 | 0.795 | 0.790 | +0.005 |
| disp | 1.47e+01 | 1.00e+00 | 0.230 | 0.227 | +0.003 |
| barrier | 1.00e+01 | 1.47e+00 | 0.513 | 0.503 | +0.010 |

## Scope note (delta interaction)

`b` is only the ridge baseline; the residual `delta = y - b` is what MLP head learns.
The alpha that minimizes b-alone NMAE is NOT necessarily optimal for the b+delta system.
System-level alpha selection belongs to SPEC_02.