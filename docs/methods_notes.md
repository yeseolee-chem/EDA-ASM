# Methods notes

## Negative (submerged) activation barriers on qmrxn20

qmrxn20 (E2/SN2) activation energies are referenced to the reactant complex
per Stuyver, Jorner & Coley, *Sci. Data* **2023**, 10, 66
(DOI: [10.1038/s41597-023-01977-8](https://doi.org/10.1038/s41597-023-01977-8));
a substantial fraction of these reactions therefore exhibit genuine
negative (submerged) barriers. These are physical, not label errors.

Concretely, in the 783-reaction locked cohort
(`outputs/v8_review/labels/labels_v9_5channel.LOCKED_783.parquet`),
`%neg(act_kcal)` is 97.5% for `qmrxn20_e2`, 85.0% for `qmrxn20_sn2`,
33.5% for `dipolar`, and 4.0% for `rgd1`. Confirmed under spec10
(barrier sign audit): the 5-channel sum reproduces the direct
`act_kcal = E(TS) − E(R)` to within 0.02 kcal/mol on
E2/SN2/rgd1, so the plotted negatives are the labelled physics.
