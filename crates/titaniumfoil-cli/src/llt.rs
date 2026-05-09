// llt.rs — Prandtl Lifting Line Theory  (Multhopp / Fourier-series formulation)
//
// Uses ODD Fourier modes only (A_1, A_3, A_5, …) which are the only modes
// that exist for a SYMMETRIC wing with symmetric loading.  Including even modes
// in a square system causes exactly-zero pivots (sin(2nθ)=0 at the centre
// collocation point), which blows up Gaussian elimination in f64.
//
// Reference: Anderson, "Fundamentals of Aerodynamics" §5.3

use std::f64::consts::PI;

// ── Public types ──────────────────────────────────────────────────────────────
pub struct WingResult {
    /// y/(b/2): 0 = centreline → 1 = tip
    pub y_norm:      Vec<f64>,
    pub cl_local:    Vec<f64>,
    pub cl_elliptic: Vec<f64>,   // ideal elliptic with same CL_wing
    pub c_ratio:     Vec<f64>,   // c/c_root at each station
    pub cl_wing:     f64,
    pub cdi:         f64,
    pub span_eff:    f64,
}

// ── Solver ────────────────────────────────────────────────────────────────────
pub fn solve(
    ar:        f64,   // aspect ratio b²/S
    taper:     f64,   // c_tip/c_root  (1=rect, 0=delta)
    twist_deg: f64,   // washout at tip (deg, positive = tip lower than root)
    alpha_deg: f64,   // root angle of attack (deg)
    a0:        f64,   // 2-D lift-curve slope (1/rad)
    al0_deg:   f64,   // 2-D zero-lift angle (deg)
) -> WingResult {
    // Number of ODD modes to use.  Each requires one collocation point.
    // N=8 → modes n=1,3,5,7,9,11,13,15 — well-conditioned for any AR>2.
    const N: usize = 8;

    let al0   = al0_deg.to_radians();
    let alpha = alpha_deg.to_radians();
    let twist = twist_deg.to_radians();

    // Odd mode indices:  n_k = 2k+1  for k=0..N-1
    let modes: Vec<usize> = (0..N).map(|k| 2 * k + 1).collect();

    // Collocation points on the HALF-span: θ ∈ (0, π/2)
    // θ_m = mπ/(2N+1)  for m=1..N  →  y_m = cos(θ_m) from near-tip to near-centre
    let theta: Vec<f64> = (1..=N).map(|m| m as f64 * PI / (2 * N + 1) as f64).collect();

    // Normalised chord c(θ)/b  (b=1 throughout)
    let c_norm = |th: f64| -> f64 {
        let cr = 2.0 / (ar * (1.0 + taper));                 // c_root/b
        cr * (1.0 - (1.0 - taper) * th.cos().abs())
    };

    // Local AoA with linear washout from root to tip
    let alpha_loc = |th: f64| -> f64 { alpha - th.cos().abs() * twist };

    // ── Build N×N system  M·A = rhs ──────────────────────────────────────────
    //
    // LLT equation at each θ_m (using only odd modes n):
    //   Σ_k  A_n_k · sin(n_k · θ_m) · [1/μ_m + n_k/sin(θ_m)]  =  α(θ_m) − α_L0
    //
    // μ_m = a0 · c_norm(θ_m) / 4
    let mut mat = vec![0.0f64; N * N];
    let mut rhs = vec![0.0f64; N];

    for (row, &th) in theta.iter().enumerate() {
        let mu     = a0 * c_norm(th) / 4.0;
        let mu_inv = 1.0 / mu.max(1e-9);
        let s_inv  = 1.0 / th.sin().max(1e-9);

        for (col, &n) in modes.iter().enumerate() {
            let nf = n as f64;
            mat[row * N + col] = (nf * th).sin() * (mu_inv + nf * s_inv);
        }
        rhs[row] = alpha_loc(th) - al0;
    }

    // ── Solve ─────────────────────────────────────────────────────────────────
    let coeffs = gauss_solve(mat, rhs, N);

    // Sanity check A_1
    if !coeffs[0].is_finite() || coeffs[0].abs() > 5.0 {
        eprintln!("  LLT warning: A1={:.3e} — check AR/taper inputs", coeffs[0]);
        return zero_result(ar, taper);
    }

    // ── Wing performance ───────────────────────────────────────────────────────
    let cl_wing = PI * ar * coeffs[0];

    // CDi = π·AR · Σ n·An²  (sum over ODD modes only)
    let sum_n_an2: f64 = modes.iter().zip(coeffs.iter())
        .map(|(&n, &an)| n as f64 * an * an)
        .sum();
    let span_eff = (coeffs[0] * coeffs[0] / sum_n_an2).clamp(0.0, 1.0);
    let cdi      = cl_wing * cl_wing / (PI * ar * span_eff.max(1e-9));

    // ── Spanwise distribution (0 = centreline → 1 = tip) ─────────────────────
    const N_PTS: usize = 61;
    let cl_e_scale = cl_wing * 4.0 / (PI * ar);
    let c_root_b   = 2.0 / (ar * (1.0 + taper));  // c_root / b

    let mut y_norm      = vec![0.0; N_PTS];
    let mut cl_local    = vec![0.0; N_PTS];
    let mut cl_elliptic = vec![0.0; N_PTS];
    let mut c_ratio     = vec![0.0; N_PTS];

    for i in 0..N_PTS {
        // Evenly-spaced θ from π/2 (centre) to small value near tip
        let t = i as f64 / (N_PTS - 1) as f64;         // 0→1
        let th = PI / 2.0 * (1.0 - t) + 1e-3 * t;     // π/2 → ε

        let yn = th.cos();  // y/(b/2)
        let gamma: f64 = modes.iter().zip(coeffs.iter())
            .map(|(&n, &an)| an * (n as f64 * th).sin())
            .sum();
        let cn = c_norm(th);
        let cl = if cn > 1e-10 { 4.0 * gamma / cn } else { 0.0 };
        let cr = if c_root_b > 1e-12 { cn / c_root_b } else { 1.0 };

        y_norm[i]      = yn;
        cl_local[i]    = cl;
        cl_elliptic[i] = cl_e_scale * (1.0 - yn * yn).max(0.0).sqrt();
        c_ratio[i]     = cr;
    }

    WingResult { y_norm, cl_local, cl_elliptic, c_ratio, cl_wing, cdi, span_eff }
}

fn zero_result(_ar: f64, taper: f64) -> WingResult {
    WingResult {
        y_norm:      vec![0.0, 1.0],
        cl_local:    vec![0.0, 0.0],
        cl_elliptic: vec![0.0, 0.0],
        c_ratio:     vec![1.0, taper],
        cl_wing: 0.0, cdi: 0.0, span_eff: 0.0,
    }
}

// ── Gaussian elimination with partial pivoting ────────────────────────────────
fn gauss_solve(mut mat: Vec<f64>, mut rhs: Vec<f64>, n: usize) -> Vec<f64> {
    for col in 0..n {
        // Partial pivot — find row with largest |element| in this column
        let pivot_row = (col..n)
            .max_by(|&a, &b| mat[a*n+col].abs().partial_cmp(&mat[b*n+col].abs()).unwrap())
            .unwrap_or(col);

        if pivot_row != col {
            for c in 0..n { mat.swap(col*n+c, pivot_row*n+c); }
            rhs.swap(col, pivot_row);
        }

        let pivot = mat[col*n+col];
        if pivot.abs() < 1e-12 { continue; }  // singular/zero column → skip

        for row in (col+1)..n {
            let f = mat[row*n+col] / pivot;
            for c in col..n { mat[row*n+c] -= f * mat[col*n+c]; }
            rhs[row] -= f * rhs[col];
        }
    }

    // Back-substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let d = mat[i*n+i];
        if d.abs() < 1e-12 { x[i] = 0.0; continue; }  // zero pivot → mode is zero
        let mut s = rhs[i];
        for j in (i+1)..n { s -= mat[i*n+j] * x[j]; }
        x[i] = s / d;
    }
    x
}
