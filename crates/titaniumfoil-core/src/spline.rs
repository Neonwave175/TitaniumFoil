// spline.f → Rust (direct port, Mark Drela MIT XFOIL)
// All routines are pure functions — no global state.

/// End condition for splind / segspld.
pub enum EndCond {
    ZeroSecondDeriv,    // 999.0 in Fortran
    ZeroThirdDeriv,     // -999.0 in Fortran
    Specified(f64),     // explicit dX/dS value
}

// ── Tridiagonal solver ────────────────────────────────────────────────────────

/// Solve a tridiagonal system in-place.
/// Layout: A is diagonal, B is sub-diagonal, C is super-diagonal.
/// D (RHS) is overwritten with the solution. A and C are destroyed.
pub fn trisol(a: &mut [f64], b: &[f64], c: &mut [f64], d: &mut [f64]) {
    let n = d.len();
    for k in 1..n {
        let km = k - 1;
        c[km] /= a[km];
        d[km] /= a[km];
        a[k] -= b[k] * c[km];
        d[k] -= b[k] * d[km];
    }
    d[n - 1] /= a[n - 1];
    for k in (0..n - 1).rev() {
        d[k] -= c[k] * d[k + 1];
    }
}

// ── Spline coefficient builders ───────────────────────────────────────────────

/// Compute spline derivatives XS for X(S) with zero-2nd-derivative ends.
/// Equivalent to SPLIND with XS1=XS2=999.
pub fn spline(x: &[f64], xs: &mut [f64], s: &[f64]) {
    splind(x, xs, s, EndCond::ZeroSecondDeriv, EndCond::ZeroSecondDeriv);
}

/// Compute spline derivatives XS for X(S) with configurable end conditions.
pub fn splind(x: &[f64], xs: &mut [f64], s: &[f64], xs1: EndCond, xs2: EndCond) {
    let n = x.len();
    assert!(n >= 2);
    assert_eq!(s.len(), n);
    assert_eq!(xs.len(), n);

    let mut a = vec![0.0f64; n];
    let mut b = vec![0.0f64; n];
    let mut c = vec![0.0f64; n];

    for i in 1..n - 1 {
        let dsm = s[i] - s[i - 1];
        let dsp = s[i + 1] - s[i];
        b[i] = dsp;
        a[i] = 2.0 * (dsm + dsp);
        c[i] = dsm;
        xs[i] = 3.0 * ((x[i + 1] - x[i]) * dsm / dsp + (x[i] - x[i - 1]) * dsp / dsm);
    }

    // Left end condition
    match xs1 {
        EndCond::ZeroSecondDeriv => {
            a[0] = 2.0;
            c[0] = 1.0;
            xs[0] = 3.0 * (x[1] - x[0]) / (s[1] - s[0]);
        }
        EndCond::ZeroThirdDeriv => {
            a[0] = 1.0;
            c[0] = 1.0;
            xs[0] = 2.0 * (x[1] - x[0]) / (s[1] - s[0]);
        }
        EndCond::Specified(v) => {
            a[0] = 1.0;
            c[0] = 0.0;
            xs[0] = v;
        }
    }

    // Right end condition
    match xs2 {
        EndCond::ZeroSecondDeriv => {
            b[n - 1] = 1.0;
            a[n - 1] = 2.0;
            xs[n - 1] = 3.0 * (x[n - 1] - x[n - 2]) / (s[n - 1] - s[n - 2]);
        }
        EndCond::ZeroThirdDeriv => {
            b[n - 1] = 1.0;
            a[n - 1] = 1.0;
            xs[n - 1] = 2.0 * (x[n - 1] - x[n - 2]) / (s[n - 1] - s[n - 2]);
        }
        EndCond::Specified(v) => {
            a[n - 1] = 1.0;
            b[n - 1] = 0.0;
            xs[n - 1] = v;
        }
    }

    // Special case: 2 points, both zero-third-deriv → fall back to zero-second-deriv on right
    if n == 2 {
        if matches!(xs2, EndCond::ZeroThirdDeriv) {
            b[n - 1] = 1.0;
            a[n - 1] = 2.0;
            xs[n - 1] = 3.0 * (x[n - 1] - x[n - 2]) / (s[n - 1] - s[n - 2]);
        }
    }

    trisol(&mut a, &b, &mut c, xs);
}

/// Monotone averaging spline — non-oscillatory, uses slope averaging.
pub fn splina(x: &[f64], xs: &mut [f64], s: &[f64]) {
    let n = x.len();
    let mut xs1 = 0.0f64;
    let mut lend = true;

    for i in 0..n - 1 {
        let ds = s[i + 1] - s[i];
        if ds == 0.0 {
            xs[i] = xs1;
            lend = true;
        } else {
            let dx = x[i + 1] - x[i];
            let xs2 = dx / ds;
            if lend {
                xs[i] = xs2;
                lend = false;
            } else {
                xs[i] = 0.5 * (xs1 + xs2);
            }
            xs1 = xs2;
        }
    }
    xs[n - 1] = xs1;
}

/// Spline with discontinuity-aware segment joints (identical consecutive S values).
pub fn segspl(x: &[f64], xs: &mut [f64], s: &[f64]) {
    let n = x.len();
    assert!(s[0] != s[1],     "segspl: first input point duplicated");
    assert!(s[n-1] != s[n-2], "segspl: last input point duplicated");

    let mut iseg0 = 0;
    for iseg in 1..n - 2 {
        if s[iseg] == s[iseg + 1] {
            splind(
                &x[iseg0..=iseg], &mut xs[iseg0..=iseg], &s[iseg0..=iseg],
                EndCond::ZeroThirdDeriv, EndCond::ZeroThirdDeriv,
            );
            iseg0 = iseg + 1;
        }
    }
    splind(
        &x[iseg0..], &mut xs[iseg0..], &s[iseg0..],
        EndCond::ZeroThirdDeriv, EndCond::ZeroThirdDeriv,
    );
}

/// Segmented spline with specified end conditions.
pub fn segspld(x: &[f64], xs: &mut [f64], s: &[f64], xs1: f64, xs2: f64) {
    let n = x.len();
    assert!(s[0] != s[1],     "segspld: first input point duplicated");
    assert!(s[n-1] != s[n-2], "segspld: last input point duplicated");

    let make_cond = |v: f64| -> EndCond {
        if v == 999.0       { EndCond::ZeroSecondDeriv }
        else if v == -999.0 { EndCond::ZeroThirdDeriv }
        else                { EndCond::Specified(v) }
    };

    let mut iseg0 = 0;
    for iseg in 1..n - 2 {
        if s[iseg] == s[iseg + 1] {
            splind(
                &x[iseg0..=iseg], &mut xs[iseg0..=iseg], &s[iseg0..=iseg],
                make_cond(xs1), make_cond(xs2),
            );
            iseg0 = iseg + 1;
        }
    }
    splind(
        &x[iseg0..], &mut xs[iseg0..], &s[iseg0..],
        make_cond(xs1), make_cond(xs2),
    );
}

// ── Spline evaluation ─────────────────────────────────────────────────────────

/// Binary search: find index i such that s[i-1] <= ss < s[i].
/// Returns i in 1..n-1 (Fortran convention kept as comment; here 1-based becomes index).
fn locate(ss: f64, s: &[f64]) -> usize {
    let n = s.len();
    let mut ilow = 0usize;
    let mut i = n - 1;
    while i - ilow > 1 {
        let imid = (i + ilow) / 2;
        if ss < s[imid] { i = imid; } else { ilow = imid; }
    }
    i // s[i-1] <= ss <= s[i]
}

/// Evaluate spline X(SS).
pub fn seval(ss: f64, x: &[f64], xs: &[f64], s: &[f64]) -> f64 {
    let i = locate(ss, s);
    let ds  = s[i] - s[i - 1];
    let t   = (ss - s[i - 1]) / ds;
    let cx1 = ds * xs[i - 1] - x[i] + x[i - 1];
    let cx2 = ds * xs[i]     - x[i] + x[i - 1];
    t * x[i] + (1.0 - t) * x[i - 1] + (t - t * t) * ((1.0 - t) * cx1 - t * cx2)
}

/// Evaluate spline derivative dX/dS(SS).
pub fn deval(ss: f64, x: &[f64], xs: &[f64], s: &[f64]) -> f64 {
    let i = locate(ss, s);
    let ds  = s[i] - s[i - 1];
    let t   = (ss - s[i - 1]) / ds;
    let cx1 = ds * xs[i - 1] - x[i] + x[i - 1];
    let cx2 = ds * xs[i]     - x[i] + x[i - 1];
    (x[i] - x[i - 1] + (1.0 - 4.0 * t + 3.0 * t * t) * cx1 + t * (3.0 * t - 2.0) * cx2) / ds
}

/// Evaluate spline second derivative d2X/dS2(SS).
pub fn d2val(ss: f64, x: &[f64], xs: &[f64], s: &[f64]) -> f64 {
    let i = locate(ss, s);
    let ds  = s[i] - s[i - 1];
    let t   = (ss - s[i - 1]) / ds;
    let cx1 = ds * xs[i - 1] - x[i] + x[i - 1];
    let cx2 = ds * xs[i]     - x[i] + x[i - 1];
    ((6.0 * t - 4.0) * cx1 + (6.0 * t - 2.0) * cx2) / (ds * ds)
}

/// Curvature of a 2-D splined curve at arc length SS.
pub fn curv(ss: f64, x: &[f64], xs: &[f64], y: &[f64], ys: &[f64], s: &[f64]) -> f64 {
    let i = locate(ss, s);
    let ds = s[i] - s[i - 1];
    let t  = (ss - s[i - 1]) / ds;

    let cx1 = ds * xs[i - 1] - x[i] + x[i - 1];
    let cx2 = ds * xs[i]     - x[i] + x[i - 1];
    let xd  = x[i] - x[i - 1] + (1.0 - 4.0 * t + 3.0 * t * t) * cx1 + t * (3.0 * t - 2.0) * cx2;
    let xdd = (6.0 * t - 4.0) * cx1 + (6.0 * t - 2.0) * cx2;

    let cy1 = ds * ys[i - 1] - y[i] + y[i - 1];
    let cy2 = ds * ys[i]     - y[i] + y[i - 1];
    let yd  = y[i] - y[i - 1] + (1.0 - 4.0 * t + 3.0 * t * t) * cy1 + t * (3.0 * t - 2.0) * cy2;
    let ydd = (6.0 * t - 4.0) * cy1 + (6.0 * t - 2.0) * cy2;

    let sd = (xd * xd + yd * yd).sqrt().max(0.001 * ds);
    (xd * ydd - yd * xdd) / (sd * sd * sd)
}

/// Curvature derivative dκ/dS at arc length SS.
pub fn curvs(ss: f64, x: &[f64], xs: &[f64], y: &[f64], ys: &[f64], s: &[f64]) -> f64 {
    let i = locate(ss, s);
    let ds = s[i] - s[i - 1];
    let t  = (ss - s[i - 1]) / ds;

    let cx1  = ds * xs[i - 1] - x[i] + x[i - 1];
    let cx2  = ds * xs[i]     - x[i] + x[i - 1];
    let xd   = x[i] - x[i - 1] + (1.0 - 4.0 * t + 3.0 * t * t) * cx1 + t * (3.0 * t - 2.0) * cx2;
    let xdd  = (6.0 * t - 4.0) * cx1 + (6.0 * t - 2.0) * cx2;
    let xddd = 6.0 * cx1 + 6.0 * cx2;

    let cy1  = ds * ys[i - 1] - y[i] + y[i - 1];
    let cy2  = ds * ys[i]     - y[i] + y[i - 1];
    let yd   = y[i] - y[i - 1] + (1.0 - 4.0 * t + 3.0 * t * t) * cy1 + t * (3.0 * t - 2.0) * cy2;
    let ydd  = (6.0 * t - 4.0) * cy1 + (6.0 * t - 2.0) * cy2;
    let yddd = 6.0 * cy1 + 6.0 * cy2;

    let sd     = (xd * xd + yd * yd).sqrt().max(0.001 * ds);
    let bot    = sd * sd * sd;
    let dbot   = 3.0 * sd * (xd * xdd + yd * ydd);
    let top    = xd * ydd  - yd * xdd;
    let dtop   = xd * yddd - yd * xddd;
    (dtop * bot - dbot * top) / (bot * bot)
}

// ── Utility ───────────────────────────────────────────────────────────────────

/// Compute arc-length array S for a 2-D point sequence (X, Y).
pub fn scalc(x: &[f64], y: &[f64], s: &mut [f64]) {
    let n = x.len();
    s[0] = 0.0;
    for i in 1..n {
        let dx = x[i] - x[i - 1];
        let dy = y[i] - y[i - 1];
        s[i] = s[i - 1] + (dx * dx + dy * dy).sqrt();
    }
}

/// Invert the spline: find S such that X(S) = XI, using Newton iteration.
/// `si` is both the initial guess (input) and the result (output).
pub fn sinvrt(si: &mut f64, xi: f64, x: &[f64], xs: &[f64], s: &[f64]) {
    let si_save = *si;
    let span = s[s.len() - 1] - s[0];
    for _ in 0..10 {
        let res  = seval(*si, x, xs, s) - xi;
        let resp = deval(*si, x, xs, s);
        let ds   = -res / resp;
        *si += ds;
        if (ds / span).abs() < 1.0e-5 {
            return;
        }
    }
    eprintln!("sinvrt: spline inversion failed — returning initial guess");
    *si = si_save;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trisol_identity() {
        // Solve [2,-1,0; -1,2,-1; 0,-1,2] x = [6,3,4]
        // Manual solution: x = [7, 8, 6]
        let mut a = vec![2.0, 2.0, 2.0];
        let     b = vec![0.0, -1.0, -1.0]; // b[k] connects row k to k-1
        let mut c = vec![-1.0, -1.0, 0.0]; // c[k] connects row k to k+1
        let mut d = vec![6.0, 3.0, 4.0];
        trisol(&mut a, &b, &mut c, &mut d);
        let expected = [7.0, 8.0, 6.0];
        for i in 0..3 {
            assert!((d[i] - expected[i]).abs() < 1e-12,
                    "trisol mismatch at {i}: got {}, expected {}", d[i], expected[i]);
        }
    }

    #[test]
    fn spline_linear() {
        // A linear function should be reproduced exactly.
        let s  = vec![0.0, 1.0, 2.0, 3.0];
        let x  = vec![0.0, 2.0, 4.0, 6.0]; // x = 2*s
        let mut xs = vec![0.0; 4];
        spline(&x, &mut xs, &s);
        // All derivatives should be exactly 2.0
        for &v in &xs {
            assert!((v - 2.0).abs() < 1e-12, "derivative should be 2, got {v}");
        }
    }

    #[test]
    fn seval_linear() {
        let s  = vec![0.0, 1.0, 2.0, 3.0];
        let x  = vec![0.0, 1.0, 2.0, 3.0];
        let mut xs = vec![0.0; 4];
        spline(&x, &mut xs, &s);
        // Evaluate at several interior points
        for &ss in &[0.3, 0.7, 1.2, 1.9, 2.5] {
            let val = seval(ss, &x, &xs, &s);
            assert!((val - ss).abs() < 1e-12, "seval at {ss}: got {val}");
        }
    }

    #[test]
    fn deval_linear() {
        let s  = vec![0.0, 1.0, 2.0];
        let x  = vec![0.0, 3.0, 6.0];
        let mut xs = vec![0.0; 3];
        spline(&x, &mut xs, &s);
        for &ss in &[0.25, 0.75, 1.25, 1.75] {
            let d = deval(ss, &x, &xs, &s);
            assert!((d - 3.0).abs() < 1e-12, "deval at {ss}: got {d}");
        }
    }

    #[test]
    fn scalc_unit_square() {
        let x = vec![0.0, 1.0, 1.0, 0.0];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut s = vec![0.0; 4];
        scalc(&x, &y, &mut s);
        assert!((s[1] - 1.0).abs() < 1e-15);
        assert!((s[2] - 2.0).abs() < 1e-15);
        assert!((s[3] - 3.0).abs() < 1e-15);
    }
}
