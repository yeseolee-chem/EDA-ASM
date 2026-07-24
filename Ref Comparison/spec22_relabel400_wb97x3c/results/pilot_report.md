# spec22 G22-0 pilot — 4-check report

**Scope of this run:** the smallest of 5 planned pilot reactions
(`dipolar_000034`, 21 atoms). Serial, no MPI. `wB97X-3c EDA-NOCV TightSCF NoSym`.
16 min 25 s wall clock. `****ORCA TERMINATED NORMALLY****`.

| # | check | verdict | evidence |
|---|---|---|---|
| 1 | EDA-NOCV accepts `wB97X-3c` | **PASS** | ORCA 6.1.1 ran to normal termination with `! wB97X-3c EDA NoSym TightSCF`. |
| 2 | CP with vDZP + ECPs | **PASS** | ORCA auto-generated `eda_frag1.inp` with 11 `:(2)` ghost atoms (fragment B's basis on fragment A). Fragment SPCs completed without basis-assignment errors. |
| 3 | channel sum matches total ΔE_int | **PASS** | Pauli(+56.28) + Elstat(−9.40) + Orbital(−7.40) + Delta E⁰(XC)(−35.05) + Delta Dispersion(−4.82) = **−0.39 kcal/mol = Bond Energy**, to display precision (< 0.01 kcal/mol). |
| 4 | D4 dispersion separable | **PASS** | `Delta Dispersion −4.82 kcal/mol` printed as its own channel; not silently folded into another term. |

### Note — the XC channel
DFT EDA-NOCV in ORCA prints **five** interaction-side channels, not four: it
splits `Delta E⁰(XC)` from `Orbital Energy`. To match the paper's 4-channel
convention, fold XC into orbital (standard for DFT EDA). Both the current
BLYP labels and Espley's ds3 make the same choice.

### Not yet checked
- Full 5-reaction pilot (spec §2 requires spanning fragment-size range).
  This run covers only the smallest. Largest (94-atom TS `dipolar_003211`)
  and three intermediates still pending.
- MPI parallelism: this job ran **serial** (nprocs=1). Production at scale
  needs MPI working — the earlier `libmpi.so.40` failure was avoided by
  disabling `%pal`, not by fixing MPI. Must be resolved before 1200 jobs.

### G22-0 status
Method-level checks (1–4) all pass on a representative small case. Halting
the 5-reaction extension pending user decision on MPI setup, given the
16 min serial cost extrapolates poorly to 94-atom fragments.
