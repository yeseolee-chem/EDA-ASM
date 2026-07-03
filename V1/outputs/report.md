# V1 Claisen ASR/EDA — preflight report

Generated: 2026-05-22 (updated after ORCA install)

Spec: `V1_Claisen_ASR_EDA_Spec_v1.md` §2.2 (Environment discovery).

**STATUS: All QM tooling now available. Awaiting user go-ahead to start Phase 0
(geometry build) of the 15-substrate Claisen workflow.**

## Cluster context

- host: `gate1.hpc`
- user: `yeseo1ee`
- scheduler: SLURM (cpu1: 48c/768GB nodes, cpu2: 256c/~1TB nodes)
- conda: base env (`/home1/yeseo1ee/miniconda3`), Python 3.13.12

## ADF / AMS — OK ✅

- `AMSBIN=/home1/yeseo1ee/ams2026.103/bin`
- `SCMLICENSE=/home1/yeseo1ee/ams2026.103/license.txt`
- `$AMSBIN/ams` → AMS 2026.103, build 202605012303 (ADF + BAND + ADFGUI)
- PLAMS 2026.103 imports via `$AMSBIN/amspython`
- Activation: `source $HOME/ams2026.103/amsbashrc.sh`

→ Phase 3 (EDA single-point at ZORA-BLYP-D3(BJ)/TZ2P) is **ready to run**
   once geometries are available.

## ORCA (QM, for `wB97X-3c`) — INSTALLED ✅

- Path: `$HOME/orca6/orca_6_1_1/orca` (ORCA 6.1.1, AVX2 build)
- Activation: `source $HOME/orca6/orca-env.sh`
- OpenMPI: 4.1.5 module loaded automatically by env script (ABI-compatible with 4.1.8 build target)
- **Smoke test passed**: H₂ at 0.74 Å, `! wB97X-3c TightSCF`, `FINAL SINGLE POINT ENERGY = -1.181762398686 Eh`, 1.08 sec runtime, `ORCA TERMINATED NORMALLY`

**Install caveat — selective extraction.** GPFS user quota (~110 GB) precluded
full ORCA install (~8 GB). Extracted only DFT/SCF/geom/freq binaries
(~2.4 GB). Excluded (recoverable later if quota increases):
- `autoci_*` — automated CI / CC implementation
- `orca_mrcc*` — MRCC interface
- `orca_mp2*`, `orca_dlpno*`, `orca_ccsd*`, `orca_ccpt*` — post-HF
- `orca_eprnmr*`, `orca_pnmr*` — NMR
- `orca_casscf*`, `orca_nevpt*` — multireference
- `orca_eda*` — ORCA EDA (we use ADF for EDA anyway)
- `orca_esd*`, `orca_mcrpa*`, `orca_fitpes*`, `orca_magrelax*` — specialty modules

For V1 Claisen spec workflow (`! wB97X-3c Opt Freq` + relaxed scan + OptTS),
none of the excluded modules are needed.

→ Phase 1 (ORCA geometry: reactant opt+freq, TS scan, TS OptTS+freq) is
**ready to run**.

### Below — original diagnostic at first preflight (kept for audit) ###

> _The text below documents the missing-ORCA state before the install
> was completed. Kept for reproducibility audit._

## ORCA (QM, for `wB97X-3c`) — earlier state: MISSING ❌

Re-checked thoroughly across the UBAI module system and filesystem after user
prompt to verify. **Confirmed not installed system-wide.**

Evidence:

1. **OpenHPC module tree** (`MODULEPATH=/opt/ohpc/pub/modulefiles`, what `module avail`
   shows by default): compilers (Intel 2023, GCC 9/12), MPI (OpenMPI 5, Intel MPI),
   CUDA (11.x–13.x), MKL, BLAS, R, Julia, Python 3.11, ANSYS, Lumerical. **No ORCA.**

2. **UBAI app module tree** (`/gpfs/TGM/local/etc/modulefiles`, the legacy `module`
   path shown in the UBAI 접속 가이드 v2.0.2): only `ANACONDA, Compiler, CUDA,
   GNUcompiler, GROMACS, MKL, MPI, NV_HPC_SDK, QE`. **No ORCA.** Note QE
   (Quantum ESPRESSO) is here — periodic plane-wave DFT, not usable for gas-phase
   Claisen TS, so it cannot substitute.

3. **App install dirs** (`/gpfs/TGM/Apps`): `ANACONDA, CUDA, GNUcompiler, gpu-burn,
   GROMACS, MELLANOX, NAMD, NVHPC, QE`. **No ORCA.**

4. **lmod hidden / spider search**: `module --show-hidden avail`, `module spider orca`,
   `module spider ORCA`, `module keyword orca` — all return nothing.

5. **Filesystem-wide find** for `orca*` under `/gpfs`: only matches are the GNOME
   accessibility screen reader's autostart files (one `.desktop` per compute node
   under `/gpfs/TGM/NODES/n*/etc/xdg/autostart/orca-autostart.desktop`). The
   `/usr/bin/orca` Python script (v3.28.2) on the gate is the same GNOME tool.

6. **AMS install** ships only ORCA *example inputs* under
   `~/ams2026.103/examples/quild/`, not the executable.

So ORCA QM is genuinely absent. Available QM/MM stack on UBAI is: GROMACS, NAMD,
QE — none of which support `wB97X-3c` on a 15-atom organic TS the way spec §3.2
requires.

Per spec §1 CRITICAL CONSTRAINT #4:
> "ORCA/ADF 가용성을 가정하지 말 것. §2.2 preflight를 먼저 돌리고, 없으면
> **계산을 시작하지 말고 사용자에게 정확히 무엇이 없는지 보고**한다."

And the spec hard-rules out using ADF for `wB97X-3c` (CONSTRAINT #1).

### What the user needs to provide

ORCA **6.0+** (the first release that ships `wB97X-3c` natively, DOI 10.1063/5.0133026).

Acceptable forms — any one of:

1. An Environment Modules name to load on `gate1.hpc` (e.g. `module load orca/6.x`).
2. An absolute path to the ORCA install root so we can set `ORCA_DIR` and
   prepend `$ORCA_DIR` to `PATH`. ORCA needs OpenMPI of the matching ABI
   (typically OpenMPI 4.1.x) reachable via `LD_LIBRARY_PATH`.
3. Permission + a target install path so we can fetch ORCA 6 from
   `https://orcaforum.kofo.mpg.de/` (academic license, free registration).
   Note the user agreement requires the user account to download.

## Python stack (active conda base)

| package  | version    | status |
|----------|------------|--------|
| rdkit    | 2026.03.1  | OK ✅  |
| numpy    | 1.26.4     | OK ✅  |
| pandas   | 3.0.2      | OK ✅  |
| pyarrow  | 24.0.0     | OK ✅  |
| scm.plams| (base env) | MISSING — use `$AMSBIN/amspython` instead |
| d2af     | —          | not importable → manual fragmentation per spec §6 Phase 2 |

> Two Python paths: geometry build (`build_geometries.py` → RDKit) runs in
> base conda; PLAMS-driven jobs run via `$AMSBIN/amspython`. Spec §6 Phase 3
> already assumes the latter.

## SLURM queues for QM (CPU)

| Partition | Nodes | Cores/node | Mem/node | Note               |
|-----------|-------|------------|----------|--------------------|
| `cpu1`    | 10    | 48         | 768 GB   | 7 idle at probe    |
| `cpu2`    | 10    | 256        | ~1 TB    | 2 idle at probe    |

Max walltime per spec §2.1: 48 h. Both partitions are CPU-only (QM
calculations must NOT go to gpu1–gpu6 per spec §1 CONSTRAINT #5).

## What I did NOT do yet (intentional, per spec §2.2 STOP rule)

- **Phase 0 (geometry build)** — would create `runs/<id>/build/` for 15 substrates.
  Per V1 Claisen spec §2.2, I stop after preflight passes and wait for user
  before launching production.
- Phase 1 (ORCA reactant/TS opt+freq) — depends on Phase 0
- Phase 3 (ADF EDA) — depends on Phase 1
- No SLURM submission

## Quota note (GPFS)

User GPFS quota is at ~110 GB (filesystem free is 271 TB but per-user cap
is enforced cluster-wide). Current usage ~106 GB after ORCA install.
Headroom is small (~4 GB) — sufficient for the 15 ORCA Opt+Freq + 15 OptTS
runs (each generates ~50 MB of binary scratch under `runs/<id>/orca/`),
but the user should consider requesting a quota increase for any larger
batch. See cleanup options that were applied (vscode + conda caches).

## Next action — awaiting user

Spec §2.2 closing rule: do not start QM compute before user reviews this
report. To proceed, user says "Phase 0 시작" (or similar). I will then:

1. Phase 0 — build 15 RDKit geometries → `runs/<id>/build/mol.xyz` + `atom_map.json`
2. Phase 1 — emit ORCA inputs (`reactant.inp`, `ts_scan.inp`, `ts.inp`) and Slurm submit scripts (cpu1 partition, 16 cores/job, 48h walltime)
3. Smoke first substrate (e.g. `h` — unsubstituted) end-to-end before submitting the rest
4. Phase 3 — ADF EDA setup (manual fragmentation per spec §6 since d2af unavailable)


## Phase 4 assembly — 2026-05-23T04:47:35.384070+00:00

Rows: 15.  Status breakdown:



## Phase 4 assembly — 2026-05-23T04:48:25.489369+00:00

Rows: 15.  Status breakdown:

- `OK`: 15

### ASR table (kcal/mol)

```
  id  sigma_p  dE_barrier_wb97x3c  dE_strain  dV_elst  dE_Pauli    dE_oi  dE_disp  imag_freq_ts_cm1 status
nme2   -0.830             +29.281    +18.146  -78.670  +168.400 -114.640   -9.190          -603.000     OK
 nh2   -0.660             +30.787    +16.717  -75.250  +162.220 -112.540   -7.700          -598.930     OK
  oh   -0.370             +25.170    +15.640  -68.460  +148.400 -104.350   -7.430          -569.360     OK
 ome   -0.270             +30.499    +15.556  -67.750  +145.710 -104.820   -8.100          -584.580     OK
  me   -0.170             +30.493    +14.078  -77.760  +167.670 -118.860   -8.160          -631.020     OK
  ph   -0.010             +29.710    +15.309  -82.000  +176.030 -123.440   -9.490          -631.390     OK
   h   +0.000             +40.542    +12.349  -60.600  +132.740  -97.980   -7.050          -610.340     OK
   f   +0.060             +24.045    +14.063  -68.030  +147.910 -108.100   -7.080          -565.930     OK
   i   +0.180             +24.626    +13.208  -67.780  +148.770 -108.220   -9.050          -561.650     OK
  br   +0.230             +24.340    +13.805  -68.160  +148.910 -109.030   -8.630          -564.290     OK
  cl   +0.230             +24.676    +13.939  -69.380  +150.890 -110.400   -8.290          -572.210     OK
  ac   +0.500             +37.752    +13.854  -63.190  +137.290 -101.640   -8.720          -605.940     OK
 cf3   +0.540             +36.349    +13.330  -56.820  +123.120  -95.710   -8.330          -569.900     OK
  cn   +0.660             +35.973    +11.838  -55.420  +120.040  -94.790   -8.140          -561.430     OK
 no2   +0.780             +25.590    +15.142  -67.020  +144.870 -112.110   -8.530          -556.430     OK
```

### Spec §8 sanity checks

1. TS qualification: all n_imag_ts == 1 → True
2. Consistency |strain+int - frag_barrier_eda| < 0.5: max diff = 0.0000
3. Imaginary freq range: -631 to -556 cm⁻¹
