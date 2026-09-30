# D1 validation — interim report (priority 0–2)

| check | value | result |
|---|---|---|
| A3 (V5 replay) | 24/24 ok, max \|Δ\| 0.0127 kcal/mol | PASS |
| A4 (V5 port) | 24/24 agree | PASS |
| A6(a) same input | max \|ΔE\| 0.000 Eh, labels 0 | PASS |
| A6(b) nprocs 1 vs 4 | max \|ΔE\| 1.37e-09 Eh | PASS |
| A7 (V8) | 8/8 detected | PASS |
| A8 (V9) | ok hard fails 0, foreign {'3090': True, '3766': True, '4252': True}, no-bond {'3400': True, '5783': True} | PASS |
| V2a | {"L0": {"n": 24, "median_max": 0.00047401607849999997, "max_max": 0.000732416558, "median_rms": 0.0001547361893243665}, "L1": {"n": 24, "median_max": 0.000304818352, "max_max": 0.000607245417, "median_rms": 8.836171012860715e-05}, "L2": {"n": 24, "median_max": 0.000293881664, "max_max": 0.000579414607, "median_rms": 7.92033019786912e-05}} | diagnostic |

**STOP conditions (§8 row 1): none**
