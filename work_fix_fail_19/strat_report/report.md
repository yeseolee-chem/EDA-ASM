# Stratification report

- Total reactions classified: **500**
- Usable (PASS / WARN / PENDING_REVALIDATE): **461**
- Dropped (EXCLUDED / FAIL): **39**
- Thin-cell threshold: usable < **15**


## Cross-tab — source_subset (overall)

| source_subset | total | usable | dropped |
|---|---|---|---|
| Halogen | 245 | 227 | 18 |
| T1x | 255 | 234 | 21 |

## Cross-tab — source_subset × heavy_atom_count

| source_subset | heavy_atom_count | total | usable | dropped |
|---|---|---|---|---|
| Halogen | 6 | 33 | 33 | 0 |
| T1x | 5 | 80 | 72 | 8 |
| T1x | 6 | 82 | 75 | 7 |
| T1x | 7 | 93 | 87 | 6 |
| Halogen | 7 | 104 | 96 | 8 |
| Halogen | 8 | 108 | 98 | 10 |

## Cross-tab — source_subset × halogen_label

| source_subset | halogen_label | total | usable | dropped |
|---|---|---|---|---|
| Halogen | Cl | 80 | 72 | 8 |
| Halogen | F | 81 | 77 | 4 |
| Halogen | Br | 84 | 78 | 6 |
| T1x | none | 255 | 234 | 21 |

## Cross-tab — source_subset × has_sulfur

| source_subset | has_sulfur | total | usable | dropped |
|---|---|---|---|---|
| Halogen | True | 69 | 62 | 7 |
| Halogen | False | 176 | 165 | 11 |
| T1x | False | 255 | 234 | 21 |

## Cross-tab — source × heavy × halogen (finest)

| source_subset | heavy_atom_count | halogen_label | total | usable | dropped |
|---|---|---|---|---|---|
| Halogen | 6 | Br | 10 | 10 | 0 |
| Halogen | 6 | Cl | 10 | 10 | 0 |
| Halogen | 6 | F | 13 | 13 | 0 |
| Halogen | 8 | Cl | 30 | 26 | 4 |
| Halogen | 7 | F | 29 | 27 | 2 |
| Halogen | 7 | Br | 35 | 33 | 2 |
| Halogen | 8 | Br | 39 | 35 | 4 |
| Halogen | 7 | Cl | 40 | 36 | 4 |
| Halogen | 8 | F | 39 | 37 | 2 |
| T1x | 5 | none | 80 | 72 | 8 |
| T1x | 6 | none | 82 | 75 | 7 |
| T1x | 7 | none | 93 | 87 | 6 |

**⚠ 3 thin cell(s) (usable < 15):**

- source_subset=Halogen, heavy_atom_count=6, halogen_label=Br → usable=10, dropped=0
- source_subset=Halogen, heavy_atom_count=6, halogen_label=Cl → usable=10, dropped=0
- source_subset=Halogen, heavy_atom_count=6, halogen_label=F → usable=13, dropped=0

## Cross-tab — source × heavy × has_sulfur (S-risk grid)

| source_subset | heavy_atom_count | has_sulfur | total | usable | dropped |
|---|---|---|---|---|---|
| Halogen | 6 | True | 10 | 10 | 0 |
| Halogen | 6 | False | 23 | 23 | 0 |
| Halogen | 7 | True | 30 | 26 | 4 |
| Halogen | 8 | True | 29 | 26 | 3 |
| Halogen | 7 | False | 74 | 70 | 4 |
| Halogen | 8 | False | 79 | 72 | 7 |
| T1x | 5 | False | 80 | 72 | 8 |
| T1x | 6 | False | 82 | 75 | 7 |
| T1x | 7 | False | 93 | 87 | 6 |

**⚠ 1 thin cell(s) (usable < 15):**

- source_subset=Halogen, heavy_atom_count=6, has_sulfur=True → usable=10, dropped=0

## Dropped reactions — provenance

| reaction_id | source_subset | heavy_atom_count | halogen_label | has_sulfur | verdict |
|---|---|---|---|---|---|
| Halogen_BrC4H5N2_rxn10147 | Halogen | 7 | Br | False | FAIL |
| Halogen_C4ClH5N2_rxn12941 | Halogen | 7 | Cl | False | EXCLUDED |
| Halogen_C4ClH5N2_rxn12962 | Halogen | 7 | Cl | False | EXCLUDED |
| Halogen_C5FH6N_rxn16456 | Halogen | 7 | F | False | FAIL |
| Halogen_BrC4H4NS_rxn10113 | Halogen | 7 | Br | True | EXCLUDED |
| Halogen_C4ClH4NS_rxn12886 | Halogen | 7 | Cl | True | FAIL |
| Halogen_C4ClH4NS_rxn12917 | Halogen | 7 | Cl | True | FAIL |
| Halogen_C5FH5S_rxn16443 | Halogen | 7 | F | True | EXCLUDED |
| Halogen_BrC5H6NO_rxn11224 | Halogen | 8 | Br | False | FAIL |
| Halogen_BrC5H6NO_rxn11417 | Halogen | 8 | Br | False | FAIL |
| Halogen_BrC6H8N_rxn12472 | Halogen | 8 | Br | False | FAIL |
| Halogen_BrC6H8N_rxn12539.json | Halogen | 8 | Br | False | FAIL |
| Halogen_C5ClH6NO_rxn15124 | Halogen | 8 | Cl | False | FAIL |
| Halogen_C5ClH6NO_rxn15200 | Halogen | 8 | Cl | False | FAIL |
| Halogen_C5FH7N2_rxn17195 | Halogen | 8 | F | False | FAIL |
| Halogen_C5ClH6NS_rxn15438.json | Halogen | 8 | Cl | True | FAIL |
| Halogen_C5FH6NS_rxn16994 | Halogen | 8 | F | True | FAIL |
| Halogen_C6ClH7S_rxn18097 | Halogen | 8 | Cl | True | FAIL |
| T1x_C3H4O2_rxn00749 | T1x | 5 | none | False | FAIL |
| T1x_C3H4O2_rxn00752 | T1x | 5 | none | False | FAIL |
| T1x_C3H5NO_rxn01048.json | T1x | 5 | none | False | FAIL |
| T1x_C3H6O2_rxn01412.json | T1x | 5 | none | False | FAIL |
| T1x_C3H6O2_rxn01416.json | T1x | 5 | none | False | FAIL |
| T1x_C4H10O_rxn01690.json | T1x | 5 | none | False | FAIL |
| T1x_C4H3N_rxn01795 | T1x | 5 | none | False | FAIL |
| T1x_C5H4_rxn05756 | T1x | 5 | none | False | FAIL |
| T1x_C3H3NO2_rxn00420 | T1x | 6 | none | False | FAIL |
| T1x_C4H5NO_rxn02463.json | T1x | 6 | none | False | FAIL |
| T1x_C4H8O2_rxn04401.json | T1x | 6 | none | False | FAIL |
| T1x_C5H8O_rxn07246 | T1x | 6 | none | False | FAIL |
| T1x_C5H8O_rxn07305 | T1x | 6 | none | False | FAIL |
| T1x_C6H12_rxn08626.json | T1x | 6 | none | False | FAIL |
| T1x_C6H12_rxn08646.json | T1x | 6 | none | False | FAIL |
| T1x_C4H6O3_rxn03352.json | T1x | 7 | none | False | FAIL |
| T1x_C5H12O2_rxn05695 | T1x | 7 | none | False | FAIL |
| T1x_C5H3NO_rxn05742 | T1x | 7 | none | False | FAIL |
| T1x_C5H6O2_rxn06285.json | T1x | 7 | none | False | FAIL |
| T1x_C6H13N_rxn08886.json | T1x | 7 | none | False | FAIL |
| T1x_C7H10_rxn09639.json | T1x | 7 | none | False | FAIL |

## Action items

Cells with **usable < 15** found in 2 cut(s). Replacement candidates should target these strata (T1x_ subset preferred, and run cheap endpoint pre-check before D2AF + ADF EDA):

- **source_subset=Halogen**, **heavy_atom_count=6**, **halogen_label=Br** → usable=10
- **source_subset=Halogen**, **heavy_atom_count=6**, **halogen_label=Cl** → usable=10
- **source_subset=Halogen**, **heavy_atom_count=6**, **halogen_label=F** → usable=13
- **source_subset=Halogen**, **heavy_atom_count=6**, **has_sulfur=True** → usable=10
