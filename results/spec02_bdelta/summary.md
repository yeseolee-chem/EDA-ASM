# SPEC_02 — b / δ decomposition (m3, 787-rxn cohort)

3 variants under matched split + HP:
- M_b:      5 fold-avg (analytic ridge α=1)
- M_δ:      25 cells
- M_bδ:     25 cells

## Mean NMAE across cells

| channel | M_b | M_δ | M_bδ |
|---|---|---|---|
| strain | 0.835 ± 0.032 | 0.797 ± 0.065 | 0.798 ± 0.051 |
| Pauli | 0.879 ± 0.046 | 0.691 ± 0.049 | 0.755 ± 0.051 |
| V_elst | 0.909 ± 0.049 | 0.772 ± 0.082 | 0.765 ± 0.068 |
| oi | 0.904 ± 0.096 | 0.705 ± 0.043 | 0.797 ± 0.051 |
| disp | 0.286 ± 0.015 | 0.397 ± 0.218 | 0.286 ± 0.015 |
| barrier | 0.538 ± 0.031 | 0.844 ± 0.151 | 0.567 ± 0.036 |

## Cancellation ρ (barrier)

- M_b: ρ = 0.083
- M_delta: ρ = 0.133
- M_bdelta: ρ = 0.098