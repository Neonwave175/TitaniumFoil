# TitaniumFoil

GPU-accelerated NACA airfoil analysis and optimization tool for Apple Silicon, written in Rust.

TitaniumFoil is a port and reimplementation of Mark Drela's XFOIL Fortran code. It uses Apple's
Metal compute shaders to assemble the O(N²) panel influence matrix on the GPU, giving a 3–8×
speed-up over the CPU path on M-series hardware. The optimizer searches NACA 4-digit and
6-series families using differential evolution to maximize L/D across multiple Reynolds numbers.

---

## Features

- Inviscid panel method (constant-vortex panels with Kutta condition and sharp-TE bisector)
- Viscous drag via integral boundary-layer: Thwaites laminar → Michel transition → Head turbulent + Ludwieg-Tillmann Cf
- NACA 4-digit, 5-digit, and 6-series airfoil geometry generators
- Prandtl lifting line theory (Multhopp/Fourier, odd-mode Fourier for symmetric wings)
- Metal GPU acceleration on Apple Silicon (zero-copy unified memory, f32 kernel, f64 CPU finish)
- Batched GPU dispatch for the optimizer (one `wait_until_completed` per population generation)
- Designation cache: identical airfoils are evaluated once and reused across DE generations
- Reynolds number calculator with ISA atmosphere and Sutherland viscosity
- Terminal Cp distribution and polar sweep plots (ANSI colour, no dependencies)

---

## Requirements

| Requirement | Version |
|-------------|---------|
| macOS | 13 Ventura or later |
| Hardware | Apple Silicon (M1 / M2 / M3 / M4 family) |
| Xcode Command Line Tools | any recent version (for `xcrun metal`) |
| Rust toolchain | 1.70 or later |

Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

Install Xcode CLT: `xcode-select --install`

---

## Build

```sh
cargo build --release
```

The build script (`crates/titaniumfoil-metal/build.rs`) calls `xcrun metal` and
`xcrun metallib` to compile the Metal shader and embed the path in the binary via
`METAL_SHADER_DIR`. No manual shader compilation step is needed.

Binaries are placed in `target/release/`:

| Binary | Crate |
|--------|-------|
| `titaniumfoil` | `titaniumfoil-cli` |
| `titaniumfoil-opt` | `titaniumfoil-optimizer` |
| `titaniumfoil-re` | `titaniumfoil-optimizer` (bin) |
| `titaniumfoil-bench` | `titaniumfoil-cli` (bin) |

---

## Usage

### `titaniumfoil` — airfoil analysis

**Single-point mode**

```sh
titaniumfoil <naca> <alpha> [re] [nside]
```

```sh
# NACA 4412 at 4° AoA, Re=200k, default panel density
titaniumfoil 4412 4.0 200000

# NACA 63-412 at 6°, Re=1M, 121 panels per side
titaniumfoil 63412 6.0 1000000 121
```

Output: CL, CM, CDP (pressure drag), CDF (friction drag), CD, L/D, and a terminal −Cp plot.

**Polar sweep (aseq mode)**

```sh
titaniumfoil <naca> aseq <a1> <a2> <da> [re] [nside]
```

```sh
# sweep -5° to 12° in 1° steps, Re=500k
titaniumfoil 2412 aseq -5 12 1 500000

# NACA 65-415 sweep, high-res panels
titaniumfoil 65415 aseq -2 8 0.5 300000 90
```

Output: table of α, CL, CM, CDP, CDF, L/D, followed by a CL–α and L/D–α plot.

**Wing mode (lifting line theory)**

```sh
titaniumfoil <naca> wing <AR> [taper] [twist_deg] [alpha] [re] [nside]
```

```sh
# Rectangular wing, AR=6, 4° AoA
titaniumfoil 2412 wing 6.0

# Tapered wing, AR=8, taper=0.4, 2° washout, 5° AoA
titaniumfoil 4412 wing 8.0 0.4 2.0 5.0 200000
```

Output: CL_wing, CDi, span efficiency, spanwise CL distribution vs elliptic reference.

**Flags**

```
--no-plot    suppress terminal plots (useful when piping output)
```

---

### `titaniumfoil-opt` — airfoil optimizer

Interactive prompt-driven optimizer. Run with no arguments:

```sh
titaniumfoil-opt
```

Prompts (all have defaults, press Enter to accept):

| Prompt | Default | Description |
|--------|---------|-------------|
| `[0] airfoil family` | `4` | `4` = NACA 4-digit, `6` = NACA 6-series, `m` = search both |
| `[1] reynolds numbers` | `80000, 150000, 300000` | Comma-separated Re values to average L/D across |
| `[2] aoa sweep start/end/step` | `-5 / 5 / 2` | Angle-of-attack range for L/D integration |
| `[3] search bounds` | family-dependent | Min/max for each DE parameter (skipped in mixed mode) |
| `[4] optimizer: max iterations` | `100` | DE generation limit |
| `[4] optimizer: population size` | `5` | Population = popsize × n_dims |
| `[4] optimizer: random seed` | `42` | Reproducible runs |
| `[5] L/D cap` | `60` | Values above this are treated as convergence artifacts |
| `[6] panel resolution` | `65` | Points per surface side; N = 2×nside − 1 |
| `[7] convergence tolerance` | `0.01` | Stop when std/|mean| ≤ tol; 0 = run all iterations |
| `[8] CM penalty weight` | `20` | score = L/D − w×|CM|; penalises extreme camber |

The optimizer prints each candidate airfoil with per-Re L/D scores, then a generation summary
with eval/s throughput and the current best designation.

---

### `titaniumfoil-re` — Reynolds number calculator

```sh
titaniumfoil-re
```

Interactive calculator. Prompts for altitude (m), chord length and unit (mm/cm/m/in), and one or
more airspeeds with unit (m/s, km/h, mph, kts). Uses the ISA atmosphere model (troposphere
0–11 000 m, isothermal layer to 20 000 m) with Sutherland's viscosity law.

Output includes temperature, density, kinematic viscosity, speed of sound, Mach number, and
a ready-to-paste Reynolds number list for the optimizer prompt `[1]`.

---

### `titaniumfoil-bench` — benchmark

```sh
titaniumfoil-bench [naca] [alpha] [re]
```

```sh
# Default: NACA 0012, 5°, Re=1M
titaniumfoil-bench

# Custom
titaniumfoil-bench 4412 6.0 500000
```

Runs CPU and GPU panel solves at panel densities N = 129, 179, 241, 299, 359, prints a
comparison table (ms/solve, speedup, ΔCL), then runs a 21-point polar sweep at maximum density
and reports total and per-point timing.

---

## Physics

### Panel method

The inviscid solver is a constant-vortex-density panel method ported from XFOIL's `xpanel.f`.
The streamfunction `Ψ` and its normal gradient at each panel collocation point are expressed as
a linear combination of the vortex strengths `γ[i]` and the freestream. This yields an N+1 × N+1
linear system (`AIJ`·`γ` = `rhs`):

- **PSILIN** — inner kernel; computes dΨ/dγ and dΨ/dσ for one evaluation point against all panels
- **GGCALC** — assembles the full AIJ matrix by calling PSILIN for each collocation point; adds the Kutta row (`γ[0] + γ[N-1] = 0`) and, for sharp trailing edges, a bisector velocity row
- **GGCALC_FINISH** — LU-factors AIJ and back-substitutes both freestream basis solutions (α=0° and α=90°)
- **SPECAL** — superposes the two basis solutions for the actual angle of attack; calls CLCALC to integrate Cp for CL, CM, and CDP

On the GPU path, PSILIN's O(N²) work is replaced by a Metal compute kernel; GGCALC_FINISH and
SPECAL remain on the CPU.

### Boundary layer

Skin-friction drag is integrated over both surfaces (LE → TE) using a three-stage model:

1. **Thwaites laminar** — closed-form momentum integral θ²(s) = (0.45/Re_c) / U_e^6 · ∫₀ˢ U_e^5 ds; friction via the Thwaites l(λ) correlation
2. **Michel's transition criterion** — Re_θ ≥ 1.174·(1 + 22400/Re_x)·Re_x^0.46 triggers the switch to turbulent
3. **Head's entrainment** — ODE for H₁·θ; shape factor H recovered via Head's inverse correlation; **Ludwieg-Tillmann** Cf = 0.246·10^(−0.678H)·Re_θ^(−0.268)

### Lifting Line Theory

Wing analysis uses Prandtl's lifting line with a Multhopp/Fourier-series solution. Only odd
Fourier modes (A₁, A₃, A₅, …) are used; even modes produce zero-pivot singularities at the
symmetry plane for symmetric wings. N=8 odd modes (modes 1, 3, 5, … 15) are solved at N
half-span collocation points. The 2-D section lift-curve slope and zero-lift angle are taken
from two panel-method evaluations at 0° and 5°.

### NACA geometry

- **4-digit** — exact polynomial thickness form (NACA TR-460), standard parabolic-arc camber line; cosine-bunched x distribution
- **5-digit** — tabulated leading-edge parameters for series 210–250
- **6-series** — 4-digit polynomial thickness (engineering approximation); a=1.0 uniform-loading mean line from thin-airfoil theory: yc = (cli/4π)[-(1-x)ln(1-x) - x·ln(x)]

### Optimizer

Differential evolution variant **DE/rand/1/bin** with F=0.8, CR=0.7. The objective is mean L/D
averaged over the Reynolds number list and AoA sweep, with an optional CM penalty:

```
score = mean_α(CL / (CDP + CDF)) − w·|CM|
```

A designation cache (HashMap keyed on the NACA string) stores evaluated airfoils. Re-encountered
designations are served from the cache with zero GPU work. Convergence is declared when
std(scores) / |mean(scores)| ≤ tol.

---

## Performance

The panel influence matrix kernel runs on the GPU using f32 arithmetic (sufficient for
O(N²) vortex integrals; the final LU solve uses f64). On Apple Silicon the GPU and CPU share
the same physical DRAM — geometry is written directly into shared Metal buffers with no copy,
and readback is a pointer cast plus f32→f64 widening.

Typical results on M2 (N=359 panels):

| Path | ms/solve | eval/s (optimizer) |
|------|----------|--------------------|
| CPU | ~8 ms | — |
| GPU (single) | ~1.5 ms | ~200 |
| GPU (batched) | ~0.3 ms/airfoil | ~500–800 |

The batched path amortises the Metal command-buffer overhead across up to 64 airfoils per
dispatch, which is the main optimizer hot path.

---

## License

GPL-2.0. See COPYING. Based on XFOIL by Mark Drela (MIT), which is also GPL-2.0.
