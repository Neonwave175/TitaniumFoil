# TitaniumFoil — Physics Reference

This document describes the aerodynamic models used in TitaniumFoil. The implementations are
direct ports of Mark Drela's XFOIL Fortran code unless noted otherwise.

---

## 1. Panel Method

### Overview

The inviscid solver uses a **constant-vortex-density panel method** (Hess–Smith class).
Each panel carries a linearly-varying vortex sheet strength γ(s); the system is closed by the
Kutta condition γ[0] + γ[N-1] = 0 at the trailing edge. For sharp trailing edges a bisector
velocity condition replaces the last panel row.

The flow field is expressed through the streamfunction Ψ. At each collocation point i:

```
Ψ(x_i, y_i) = Σ_j dzdg[j]·γ[j]  +  Ψ_freestream  =  Ψ₀   (body BC)
```

This gives the N+1 linear system AIJ·γ = rhs, where the extra unknown is Ψ₀ (interior
streamfunction value).

### PSILIN — inner kernel

`psilin(state, io, xi, yi, nxi, nyi, siglin)` evaluates the streamfunction and its normal
gradient at point (xi, yi) due to all N vortex panels (and optionally source panels when
`siglin=true`). It fills `state.flow.dzdg[j]` = dΨ/dγ[j] for all j.

Kernel quantities for a panel from node j to node j+1 (in local coordinates x̄, ȳ):

```
g1 = ln r1²,   t1 = atan2(x̄1, ȳ)   (log and angle at start node)
g2 = ln r2²,   t2 = atan2(x̄2, ȳ)   (log and angle at end node)

Ψ_vortex = QOPI · [ (x̄1·g1/2 - x̄2·g2/2 + x̄2 - x̄1 + ȳ(t1-t2))·(γ_j+γ_{j+1})
                   + (...)·(γ_{j+1}-γ_j) ]
```

where QOPI = 1/(4π). The trailing-edge "half panel" uses HOPI = 1/(2π) with the TE geometry
factors `scs` (cosine component) and `sds` (sine component).

### GGCALC — matrix assembly

`ggcalc(state)` calls PSILIN for each of the N collocation points to fill rows 0..N-1 of AIJ,
then calls `ggcalc_setup_rhs` to add the Kutta row and, if the trailing edge is sharp, the
bisector row. Finally `ggcalc_finish` is called.

On the GPU path, the N×N panel rows are filled by the Metal kernel `panel_influence_2d`; only
`ggcalc_setup_rhs` and `ggcalc_finish` run on the CPU.

`ggcalc_setup_rhs` sets:
- Freestream RHS: GAMU[i] = −Q∞·y_i (basis α=0°), GAMU[i+IQX] = +Q∞·x_i (basis α=90°)
- Kutta row: AIJ[N,0] = AIJ[N,N-1] = 1
- Bisector row (sharp TE): replaces row N-1 with the normal-velocity condition at a point inside the TE wedge

### GGCALC_FINISH — LU solve

`ggcalc_finish(state)` LU-factors AIJ (Crout / partial-pivot, `ludcmp`/`baksub`) and
back-substitutes the two freestream basis solution vectors GAMU[:,0] and GAMU[:,IQX].

### SPECAL — superposition and pressure

`specal(state)` superposes the two basis solutions for angle of attack α:

```
γ[i] = cos(α)·GAMU[i,0] + sin(α)·GAMU[i,IQX]
```

and calls `clcalc` to integrate the pressure distribution:

```
Cp_i = 1 − (γ_i / Q∞)²   [incompressible; Prandtl-Glauert for M∞ > 0]

CL  = Σ_i Cp_avg · [(x_{i+1}−x_i)cosα + (y_{i+1}−y_i)sinα]
CM  = −Σ_i (Cp moment arm terms)   [about x/c = 0.25]
CDP = −Σ_i Cp_avg · [(y_{i+1}−y_i)cosα − (x_{i+1}−x_i)sinα]
```

---

## 2. Boundary Layer

The viscous drag model integrates over each surface separately from LE to TE. All lengths are
normalised to chord c=1, velocity to Q∞=1, so kinematic viscosity ν = 1/Re_c.

### 2.1 Thwaites laminar BL

Thwaites (1949) closed-form solution for the laminar momentum thickness θ:

```
θ²(s) = (0.45 / Re_c) · U_e^{-6} · ∫₀ˢ U_e^5 ds'
```

The integral is accumulated with the trapezoid rule. The pressure-gradient parameter:

```
λ = Re_c · θ² · (dU_e/ds)    clamped to [−0.09, 0.25]
```

The Thwaites friction function:

```
l(λ) = max(0, 0.22 + 1.57λ − 1.8λ²)
```

Local laminar skin friction and its contribution to CDF:

```
Cf = 2·l / Re_θ,    Re_θ = Re_c · U_e · θ
ΔCdf += Cf · U_e² · Δs
```

The shape factor H(λ) used at transition:

```
H = 2.61 − 3.75λ + 5.24λ²    (λ ≥ 0)
H = 2.088 + 0.0731/(λ + 0.14) (λ < 0),  clamped to [2.0, 4.0]
```

### 2.2 Michel's transition criterion

Michel (1951) empirical correlation:

```
Re_θ ≥ 1.174 · (1 + 22400/Re_x) · Re_x^{0.46}
```

where Re_x = Re_c · U_e · s. When this condition is first satisfied, the boundary layer
switches from laminar to turbulent. The turbulent initial conditions are set from the current
θ and H.

### 2.3 Head's turbulent closure

Head's entrainment method (1958) integrates two ODEs:

**Momentum integral** (von Kármán):

```
dθ/ds = Cf/2 − (H+2) · (θ/U_e) · (dU_e/ds)
```

**Entrainment** (Head's relation):

```
d(H₁·θ)/ds = 0.0306 · (H₁ − 3)^{−0.6169}
```

where H₁ is the shape factor based on the wake deficit. The Head correlations connecting H and H₁:

```
H₁ = 3.3 + 0.8234·(H − 1.1)^{−1.287}   (H ≤ 1.6)
H₁ = 3.3 + 1.5501·(H − 0.6778)^{−3.064} (H > 1.6)
```

Inverse (H₁ → H):

```
H = 1.1 + [(H₁−3.3)/0.8234]^{−1/1.287}   (H₁ ≥ 5.3, i.e. H < 1.6)
H = 0.6778 + [(H₁−3.3)/1.5501]^{−1/3.064} (H₁ < 5.3, i.e. H ≥ 1.6)
```

### 2.4 Ludwieg-Tillmann skin friction

```
Cf = 0.246 · 10^{−0.678H} · Re_θ^{−0.268}
```

This is applied at each turbulent station; the contribution to CDF is:

```
ΔCdf += Cf · U_e² · Δs
```

---

## 3. Lifting Line Theory

### Prandtl formulation

For a finite wing the spanwise lift distribution Γ(y) satisfies Prandtl's integro-differential
equation. TitaniumFoil uses the Fourier-series representation:

```
Γ(θ) = 2·b·Q∞ · Σ_n A_n · sin(n·θ)    with  y = (b/2)·cos(θ)
```

### Multhopp/Fourier system

The LLT equation at each collocation point θ_m:

```
Σ_k A_{n_k} · sin(n_k·θ_m) · [1/μ_m + n_k/sin(θ_m)] = α(θ_m) − α_{L0}
```

where μ_m = a₀·c(θ_m)/4 and a₀ is the 2-D lift-curve slope in 1/rad.

### Odd-mode-only formulation

For a **symmetric wing with symmetric loading** only odd Fourier modes exist.
Including even modes in the N×N square system causes exactly-zero diagonal pivots at the
symmetric collocation point (sin(2nθ)=0 at θ=π/2), which makes Gaussian elimination singular.

TitaniumFoil uses N=8 odd modes (n = 1, 3, 5, 7, 9, 11, 13, 15) with 8 collocation points
at θ_m = m·π/(2N+1) for m=1..N, spanning the half-span near-tip to near-centreline.

### Wing performance

```
CL_wing = π·AR·A₁

CDi     = π·AR · Σ_k n_k·A_{n_k}²

e       = A₁² / Σ_k n_k·A_{n_k}²   (Oswald span efficiency)
```

The 2-D section parameters a₀ and α_{L0} are extracted from two panel-method evaluations at
α=0° and α=5°.

---

## 4. NACA Airfoil Geometry

### 4-digit series

Thickness distribution (NACA TR-460 polynomial):

```
y_t(x) = (t/0.20) · [0.2969√x − 0.1260x − 0.3516x² + 0.2843x³ − 0.1015x⁴]
```

Camber line (parabolic arc):

```
y_c(x) = m/p²  · (2px − x²)         for x ≤ p
y_c(x) = m/(1-p)² · (1−2p + 2px − x²)  for x > p
```

where m = max camber fraction, p = chordwise position of max camber.
Designation digits MPTT: m = M/100, p = P/10, t = TT/100.

### 5-digit series

Uses the same thickness form. The camber line is the three-term polynomial from Abbott &
Von Doenhoff, parameterised by mf (location of max camber) and c (camber multiplier):

```
y_c(x) = (c/6) · (x³ − 3·mf·x² + mf²·(3−mf)·x)    for x ≤ mf
y_c(x) = (c/6) · mf³ · (1−x)                          for x > mf
```

Tabulated (mf, c) pairs for design series 210, 220, 230, 240, 250.

### 6-series

**Thickness**: 4-digit polynomial form (engineering approximation; exact 6-series thickness
requires Abbott & Von Doenhoff tabular data, but the 4-digit polynomial reproduces max
thickness correctly and differences are second-order for most engineering purposes).

**Camber**: a=1.0 uniform-loading mean line (NACA TR-824 §6). Derived from thin-airfoil theory
with uniform bound vorticity Γ ∝ (1−x):

```
y_c(x) = (cli / 4π) · [−(1−x)·ln(1−x) − x·ln(x)]
```

This gives CL₀ = cli_design at α=0. Max camber is at x=0.5c (midchord).
The singularities at x=0 and x=1 resolve to y_c=0 by L'Hôpital's rule.

Designation format: NACA [6][S]-[C][TT]
- S = sub-series digit (3 → 63-series, 4 → 64-series, … 7 → 67-series)
- C = design lift coefficient × 10 (cli = C/10)
- TT = thickness / chord × 100 (t = TT/100)

All generators use a cosine-bunched x distribution (denser at LE and TE) with nside points per
side; total panels N = 2·nside − 1.

---

## 5. Optimizer

### Differential Evolution — DE/rand/1/bin

For each member i of the population of size P:

1. Choose three distinct random members r1, r2, r3 ≠ i
2. **Mutation**: v = pop[r1] + F·(pop[r2] − pop[r3])   with F=0.8
3. **Crossover** (binomial, CR=0.7): trial[dim] = v[dim] if rand < CR or dim == j_rand, else pop[i][dim]
4. **Selection**: if score(trial) > score(pop[i]), replace

Population size = popsize × n_dims (default popsize=5; n_dims=3 for 4-digit/6-series, 4 for
mixed mode). Random numbers come from a 64-bit LCG seeded by the user seed.

### Objective function

```
score = mean_{α} [ CL / (CDP + CDF) − w·|CM| ]
```

averaged over the specified Reynolds numbers and AoA sweep points. L/D values above the cap
(default 60) are discarded as convergence artifacts. Failed evaluations (non-finite CL) return
0, causing the candidate to be selected against. The CM penalty w discourages extreme camber
that would create unacceptable pitching moments in practice.

### Designation cache

A global `HashMap<String, (per_Re_scores, avg)>` stores results keyed on the NACA designation
string. Because the parameter space is discrete (rounded digit values), many DE trial vectors
map to the same designation. Cache hits require zero GPU work. The cache is never invalidated
within a run (the evaluation is deterministic for fixed config).

### Batched GPU dispatch

The optimizer uses `compute_panel_matrix_batch_gpu` which uploads geometry for an entire DE
population generation (up to 64 airfoils) in parallel (rayon), submits a single Metal command
buffer with a 2-D threadgroup grid (batch × N rows), and waits once. Readback and RHS setup
also run in parallel. This reduces the Metal overhead from P synchronisation points to 1 per
generation.

---

## References

- Drela, M. (1989). XFOIL: An analysis and design system for low Reynolds number airfoils. *Low Reynolds Number Aerodynamics*, Lecture Notes in Engineering 54. Springer.
- Thwaites, B. (1949). Approximate calculation of the laminar boundary layer. *Aeronautical Quarterly* 1:245–280.
- Michel, R. (1951). Etude de la transition sur les profils d'aile. *ONERA Rapport* 1/1578A.
- Head, M.R. (1958). Entrainment in the turbulent boundary layer. *ARC R&M* 3152.
- Ludwieg, H. & Tillmann, W. (1950). Investigations of the wall shearing stress in turbulent boundary layers. *NACA TM* 1256.
- Abbott, I.H. & Von Doenhoff, A.E. (1959). *Theory of Wing Sections*. Dover.
- Anderson, J.D. (2017). *Fundamentals of Aerodynamics*, 6th ed. McGraw-Hill, §5.3.
- Storn, R. & Price, K. (1997). Differential evolution — a simple and efficient heuristic for global optimization. *J. Global Optimization* 11:341–359.
- ICAO Doc 7488 / ISO 2533. Standard Atmosphere.
- Sutherland, W. (1893). The viscosity of gases and molecular force. *Phil. Mag.* 5(36):507.
