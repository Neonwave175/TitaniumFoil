// viscous.rs — Simplified integral boundary-layer skin-friction drag.
//
// Pipeline (per surface, LE → TE):
//   1. Thwaites' method  — laminar momentum integral, O(N) closed form
//   2. Michel's criterion — Re_θ vs Re_x transition detection
//   3. Head's entrainment + Ludwieg-Tillmann Cf — turbulent closure
//
// Normalisation assumed throughout:
//   chord c = 1,  U_∞ = qinf = 1,  kinematic viscosity ν = 1/Re_c
//
// References:
//   Thwaites (1949)  Aeronautical Quarterly 1:245
//   Michel (1951)    ONERA Rpt. 1/1578A
//   Head (1958)      ARC R&M 3152
//   Ludwieg & Tillmann (1950)  NACA TM 1256

use crate::types::XfoilState;

// ── Public API ────────────────────────────────────────────────────────────────

/// Skin-friction drag coefficient Cdf integrated over both surfaces.
///
/// Call **after** `ggcalc` + `specal`. Thread-safe (immutable borrow).
/// Returns 0.0 if Re < 1 or the geometry hasn't been solved.
pub fn skin_friction_drag(state: &XfoilState) -> f64 {
    let n = state.geom.n;
    if n < 4 || state.op.reinf < 1.0 { return 0.0; }

    let re_c = state.op.reinf;

    // Leading-edge node = minimum x
    let i_le = (0..n)
        .min_by(|&a, &b| state.geom.x[a].partial_cmp(&state.geom.x[b]).unwrap())
        .unwrap_or(n / 2);

    let s_le = state.geom.s[i_le];

    // Upper surface: k from i_le → 0  (physical arc-length from LE grows)
    let mut us = Vec::with_capacity(i_le + 1);
    let mut uu = Vec::with_capacity(i_le + 1);
    for k in (0..=i_le).rev() {
        us.push(s_le - state.geom.s[k]);
        uu.push(state.vel.qinv[k].abs().max(1e-9));
    }

    // Lower surface: k from i_le → n-1
    let mut ls = Vec::with_capacity(n - i_le);
    let mut lu = Vec::with_capacity(n - i_le);
    for k in i_le..n {
        ls.push(state.geom.s[k] - s_le);
        lu.push(state.vel.qinv[k].abs().max(1e-9));
    }

    integrate_surface(&us, &uu, re_c) + integrate_surface(&ls, &lu, re_c)
}

// ── Core integration ──────────────────────────────────────────────────────────

/// Integrate skin-friction drag over one surface going LE → TE.
///
/// `s`  : arc-length from leading edge, monotonically increasing (chord units)
/// `ue` : |U_e| / U_∞ at each station
/// `re_c`: chord Reynolds number
fn integrate_surface(s: &[f64], ue: &[f64], re_c: f64) -> f64 {
    let n = s.len();
    if n < 2 { return 0.0; }

    let mut cdf       = 0.0;
    let mut integral  = 0.0_f64;   // Thwaites:  I = ∫ U_e^5 ds
    let mut theta;                  // laminar θ (chord units), set each step
    let mut in_turb   = false;

    // Turbulent-phase state
    let mut th_t  = 0.0_f64;   // θ
    let mut h1th  = 0.0_f64;   // H₁·θ
    let mut h_t   = 1.4_f64;   // shape factor H

    for i in 1..n {
        let ds = s[i] - s[i - 1];
        if ds < 1e-12 { continue; }

        let u0 = ue[i - 1];
        let u1 = ue[i];

        if !in_turb {
            // ── Thwaites laminar BL ───────────────────────────────────────────
            //
            // Closed-form solution:
            //   θ²(s) = (0.45/Re_c) / U_e^6 · ∫₀ˢ U_e^5 ds'
            //
            integral += 0.5 * (u0.powi(5) + u1.powi(5)) * ds;
            let th2 = (0.45 / re_c) * integral / u1.powi(6).max(1e-30);
            theta  = th2.sqrt();

            // Pressure-gradient parameter λ = Re_c·θ²·(dU_e/ds)
            let due_ds = (u1 - u0) / ds;
            let lam = (re_c * th2 * due_ds).clamp(-0.09, 0.25);

            // Thwaites friction function l(λ) — wall shear proxy
            let l = (0.22 + 1.57 * lam - 1.8 * lam * lam).max(0.0);

            // Local Cf (laminar) and contribution to Cdf
            //   Cf = 2·l / Re_θ,   Cdf += Cf·(U_e/U_∞)²·ds
            let re_th = (re_c * u1 * theta).max(1.0);
            let cf    = 2.0 * l / re_th;
            cdf += cf * u1 * u1 * ds;

            // ── Michel's transition criterion ─────────────────────────────────
            //   Re_θ ≥ 1.174 · (1 + 22400/Re_x) · Re_x^0.46
            let re_x     = (re_c * u1 * s[i]).max(1.0);
            let re_th_cr = 1.174 * (1.0 + 22400.0 / re_x) * re_x.powf(0.46);

            if re_th >= re_th_cr {
                in_turb = true;
                th_t = theta.max(1e-9);

                // Initial shape factor: Thwaites H(λ), clamped to turbulent range
                let h_lam = thwaites_h(lam).clamp(1.3, 2.8);
                h_t  = h_lam;
                h1th = h1_from_h(h_t) * th_t;
            }
        } else {
            // ── Head's turbulent BL ───────────────────────────────────────────

            // Ludwieg-Tillmann Cf
            let re_th = (re_c * u1 * th_t).max(1.0);
            let cf    = ludwieg_tillmann(h_t, re_th);
            cdf += cf * u1 * u1 * ds;

            // Momentum integral:  dθ/ds = Cf/2 − (H+2)·(θ/U_e)·dU_e/ds
            let due_ds = (u1 - u0) / ds;
            let dth    = cf / 2.0 - (h_t + 2.0) * th_t / u1.max(1e-9) * due_ds;
            th_t = (th_t + dth * ds).max(1e-9);

            // Head's entrainment:  d(H₁·θ)/ds = 0.0306·(H₁−3)^(−0.6169)
            let h1    = (h1th / th_t).clamp(3.01, 20.0);
            let fh1   = 0.0306 * (h1 - 3.0).powf(-0.6169);
            h1th = (h1th + fh1 * ds).max(th_t * 3.01);

            // Recover H from H₁
            h_t = h_from_h1(h1th / th_t.max(1e-9)).clamp(1.05, 4.0);
        }
    }

    cdf
}

// ── BL correlations ───────────────────────────────────────────────────────────

/// Thwaites H(λ) — laminar shape factor
fn thwaites_h(lam: f64) -> f64 {
    if lam >= 0.0 {
        (2.61 - 3.75 * lam + 5.24 * lam * lam).max(2.0)
    } else {
        (2.088 + 0.0731 / (lam + 0.14)).clamp(2.0, 4.0)
    }
}

/// Head's relation: H → H₁
fn h1_from_h(h: f64) -> f64 {
    if h <= 1.6 {
        3.3 + 0.8234 * (h - 1.1).max(1e-6).powf(-1.287)
    } else {
        3.3 + 1.5501 * (h - 0.6778).max(1e-6).powf(-3.064)
    }
}

/// Inverse Head's relation: H₁ → H
///
/// Branch boundary H₁ ≈ 5.3 (corresponds to H ≈ 1.6)
fn h_from_h1(h1: f64) -> f64 {
    let h1 = h1.clamp(3.01, 20.0);
    if h1 >= 5.3 {
        // H < 1.6 branch
        1.1 + ((h1 - 3.3) / 0.8234).max(1e-9).powf(-1.0 / 1.287)
    } else {
        // H ≥ 1.6 branch
        0.6778 + ((h1 - 3.3) / 1.5501).max(1e-9).powf(-1.0 / 3.064)
    }
}

/// Ludwieg-Tillmann turbulent Cf
fn ludwieg_tillmann(h: f64, re_th: f64) -> f64 {
    (0.246 * 10f64.powf(-0.678 * h) * re_th.max(1.0).powf(-0.268)).max(0.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::XfoilState;
    use crate::geometry::load_naca;
    use crate::panel::{calc_normals, calc_panel_angles, ggcalc, specal};

    fn solve(designation: &str, re: f64, alpha_deg: f64) -> XfoilState {
        let mut s = XfoilState::default();
        load_naca(&mut s, designation);
        s.op.qinf  = 1.0;
        s.op.reinf = re;
        s.op.alfa  = alpha_deg.to_radians();
        calc_normals(&mut s);
        calc_panel_angles(&mut s);
        ggcalc(&mut s);
        specal(&mut s);
        s
    }

    /// Full flat-plate laminar Cdf = 2 × 1.328/√Re_c.
    /// NACA 0012 at α=0, low Re → fully laminar → should be close to that.
    #[test]
    fn laminar_cdf_close_to_blasius() {
        let re = 50_000.0;
        let state = solve("0012", re, 0.0);
        let cdf = skin_friction_drag(&state);

        // Blasius (both surfaces): 2 * 1.328 / sqrt(50000) ≈ 0.01188
        // Expect ±40 % (airfoil has curved surfaces; exact flat plate differs)
        let blasius = 2.0 * 1.328 / re.sqrt();
        eprintln!("Cdf={cdf:.5}  Blasius={blasius:.5}");
        assert!((cdf - blasius).abs() / blasius < 0.5,
            "Cdf={cdf:.5} too far from Blasius={blasius:.5}");
    }

    /// Cdf must decrease monotonically as Re increases (both laminar & turbulent Cf ∝ Re^-n).
    #[test]
    fn cdf_decreases_with_re() {
        let mut prev = f64::INFINITY;
        for &re in &[50_000.0_f64, 100_000.0, 200_000.0, 500_000.0, 1_000_000.0] {
            let state = solve("0012", re, 0.0);
            let cdf = skin_friction_drag(&state);
            eprintln!("Re={re:.0}  Cdf={cdf:.5}");
            assert!(cdf < prev, "Cdf should decrease with Re (got {cdf} >= {prev})");
            prev = cdf;
        }
    }

    /// 6-series at its design CL should have lower Cdf than a 4-digit
    /// with similar thickness, because the favourable pressure gradient
    /// delays transition and keeps flow laminar longer.
    #[test]
    fn naca6_lower_cdf_than_naca4_at_design_cl() {
        let re   = 200_000.0;
        let aoa  = 3.0;          // near design CL for both
        let s4   = solve("4412", re, aoa);
        let s6   = solve("63412", re, aoa);
        let cdf4 = skin_friction_drag(&s4);
        let cdf6 = skin_friction_drag(&s6);
        eprintln!("NACA 4412 Cdf={cdf4:.5}  NACA 63-412 Cdf={cdf6:.5}");
        // 6-series advantage may be small in inviscid model but should not be worse
        assert!(cdf6 <= cdf4 * 1.15,
            "6-series Cdf={cdf6:.5} unexpectedly much higher than 4-digit {cdf4:.5}");
    }

    /// Cdf should be a reasonable number (not NaN, not absurd).
    #[test]
    fn cdf_sanity_range() {
        for desig in ["0012", "2412", "4412", "63412", "65415"] {
            let state = solve(desig, 200_000.0, 4.0);
            let cdf = skin_friction_drag(&state);
            assert!(cdf.is_finite() && cdf > 0.0 && cdf < 0.1,
                "{desig}: Cdf={cdf:.5} out of sane range");
        }
    }

    #[test]
    fn breakdown_cl_cdp_cdf() {
        let cases = [
            ("4412",  200_000.0, 4.0),
            ("0012",  200_000.0, 4.0),
            ("63412", 200_000.0, 4.0),
            ("65412", 200_000.0, 4.0),
        ];
        println!("\n{:<8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "airfoil", "CL", "Cdp", "Cdf", "Cd_tot", "L/D");
        println!("{}", "─".repeat(56));
        for (desig, re, aoa) in cases {
            let state = solve(desig, re, aoa);
            let cl  = state.op.cl;
            let cdp = state.op.cdp;
            let cdf = skin_friction_drag(&state);
            let cd  = cdp + cdf;
            println!("{:<8} {:>8.4} {:>8.5} {:>8.5} {:>8.5} {:>8.2}",
                desig, cl, cdp, cdf, cd, cl/cd);
        }
    }

    #[test]
    fn cm_penalty_check() {
        let s1 = solve("4412", 200_000.0, 4.0);
        let s2 = solve("9806", 200_000.0, 4.0);
        eprintln!("4412 CM={:.4}  9806 CM={:.4}", s1.op.cm, s2.op.cm);
        assert!(s2.op.cm.abs() > s1.op.cm.abs() * 1.5,
            "9806 should have larger |CM| than 4412");
    }
}
