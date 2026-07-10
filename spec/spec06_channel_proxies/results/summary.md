# SPEC_06 - channel-matched proxies (m3 v7, 776 rxns)

- N cohort: 776
- d26/d27/d28 SCF ok: 776 / 776 / 776

## Physics sanity (pearson vs channel label)
- pearson(d26, V_elst) = +0.345
- pearson(d27, Pauli)  = +0.074
- pearson(d28, E_orb)  = +0.087


## NMAE per channel per variant (5-fold pooled OOF)

| channel | base_24d | base_25d | base_24d_d26 | base_24d_d27 | base_24d_d28 | base_24d_d26_27_28 | base_25d_d26_27_28 |
|---|---|---|---|---|---|---|---|
| strain | 0.618 | 0.543 | 0.608 | 0.592 | 0.616 | 0.595 | 0.531 |
| Pauli | 0.438 | 0.428 | 0.444 | 0.398 | 0.437 | 0.409 | 0.388 |
| elst | 0.418 | 0.419 | 0.423 | 0.403 | 0.419 | 0.402 | 0.393 |
| oi | 0.410 | 0.401 | 0.407 | 0.359 | 0.413 | 0.362 | 0.344 |
| disp | 0.182 | 0.171 | 0.180 | 0.176 | 0.181 | 0.178 | 0.169 |
| barrier | 0.506 | 0.497 | 0.523 | 0.514 | 0.537 | 0.510 | 0.456 |

## Ablation deltas vs base_24d
See ablation_deltas.csv (smaller / more negative = better).