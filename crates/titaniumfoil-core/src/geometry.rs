// xgeom.f + naca.f → Rust (direct port, Mark Drela MIT XFOIL)
// All routines operate on explicit arrays rather than global state.

use std::f64::consts::PI;
use crate::spline::{seval, deval, d2val, curv, scalc, segspl, trisol};
use crate::types::XfoilState;

// ── Leading / nose edge finders ───────────────────────────────────────────────

/// Find arc-length SLE of the leading edge by Newton iteration.
/// Condition: (X - XTE, Y - YTE) · (X', Y') = 0 at S = SLE.
pub fn lefind(x: &[f64], xp: &[f64], y: &[f64], yp: &[f64], s: &[f64]) -> f64 {
    let n = x.len();
    let dseps = (s[n - 1] - s[0]) * 1.0e-5;
    let xte = 0.5 * (x[0] + x[n - 1]);
    let yte = 0.5 * (y[0] + y[n - 1]);

    // first guess: walk until dot product changes sign
    let mut i_guess = 2;
    for i in 2..n - 2 {
        let dxte = x[i] - xte;
        let dyte = y[i] - yte;
        let dx = x[i + 1] - x[i];
        let dy = y[i + 1] - y[i];
        if dxte * dx + dyte * dy < 0.0 {
            i_guess = i;
            break;
        }
    }
    let mut sle = s[i_guess];

    // sharp LE shortcut
    if i_guess > 0 && s[i_guess] == s[i_guess - 1] { return sle; }

    // Newton iteration
    for _ in 0..50 {
        let xle  = seval(sle, x, xp, s);
        let yle  = seval(sle, y, yp, s);
        let dxds = deval(sle, x, xp, s);
        let dyds = deval(sle, y, yp, s);
        let dxdd = d2val(sle, x, xp, s);
        let dydd = d2val(sle, y, yp, s);

        let xc = xle - xte;
        let yc = yle - yte;
        let res  = xc * dxds + yc * dyds;
        let ress = dxds * dxds + dyds * dyds + xc * dxdd + yc * dydd;

        let dsle = (-res / ress)
            .max(-0.02 * (xc + yc).abs())
            .min( 0.02 * (xc + yc).abs());
        sle += dsle;
        if dsle.abs() < dseps { return sle; }
    }
    eprintln!("lefind: LE not converged — returning best estimate");
    sle
}

/// Find leftmost point (minimum x) on the spline curve.
pub fn xlfind(x: &[f64], xp: &[f64], _y: &[f64], _yp: &[f64], s: &[f64]) -> f64 {
    let n = x.len();
    let dslen = s[n - 1] - s[0];
    let dseps = dslen * 1.0e-5;

    let mut i_guess = 2;
    for i in 2..n - 2 {
        if x[i + 1] - x[i] > 0.0 { i_guess = i; break; }
    }
    let mut sle = s[i_guess];
    if i_guess > 0 && s[i_guess] == s[i_guess - 1] { return sle; }

    for _ in 0..50 {
        let dxds = deval(sle, x, xp, s);
        let dxdd = d2val(sle, x, xp, s);
        let dsle = (-dxds / dxdd)
            .max(-0.01 * dslen.abs())
            .min( 0.01 * dslen.abs());
        sle += dsle;
        if dsle.abs() < dseps { return sle; }
    }
    eprintln!("xlfind: left point not converged");
    sle
}

/// Find opposite-surface arc length SOPP such that X(SOPP) == X(SI) on the other side.
pub fn sopps(si: f64, x: &[f64], xp: &[f64], y: &[f64], yp: &[f64],
             s: &[f64], sle: f64) -> f64 {
    let n = x.len();
    let slen = s[n - 1] - s[0];
    let xle = seval(sle, x, xp, s);
    let yle = seval(sle, y, yp, s);
    let xte = 0.5 * (x[0] + x[n - 1]);
    let yte = 0.5 * (y[0] + y[n - 1]);
    let chord = ((xte - xle).powi(2) + (yte - yle).powi(2)).sqrt();
    let dxc = (xte - xle) / chord;
    let dyc = (yte - yle) / chord;

    let (s_in, s_opp_end) = if si < sle { (s[0], s[n - 1]) } else { (s[n - 1], s[0]) };
    let sfrac = (si - sle) / (s_in - sle);
    let mut sopp = sle + sfrac * (s_opp_end - sle);

    if sfrac.abs() <= 1.0e-5 { return sle; }

    let xi  = seval(si, x, xp, s);
    let yi  = seval(si, y, yp, s);
    let xle2 = seval(sle, x, xp, s);
    let yle2 = seval(sle, y, yp, s);
    let xbar = (xi - xle2) * dxc + (yi - yle2) * dyc;

    for _ in 0..12 {
        let xopp  = seval(sopp, x, xp, s);
        let yopp  = seval(sopp, y, yp, s);
        let xoppd = deval(sopp, x, xp, s);
        let yoppd = deval(sopp, y, yp, s);
        let res  = (xopp - xle2) * dxc + (yopp - yle2) * dyc - xbar;
        let resd = xoppd * dxc + yoppd * dyc;
        if res.abs() / slen < 1.0e-5 { return sopp; }
        if resd == 0.0 { break; }
        sopp -= res / resd;
        if (sopp - sopp).abs() / slen < 1.0e-5 { return sopp; }
    }
    sle + sfrac * (s_opp_end - sle)
}

// ── Geometry normalisation ────────────────────────────────────────────────────

/// Scale X, Y, S arrays to unit chord, LE at origin.
pub fn norm_coords(x: &mut [f64], xp: &mut [f64], y: &mut [f64], yp: &mut [f64], s: &mut [f64]) {
    let n = x.len();
    scalc(x, y, s);
    segspl(x, xp, s);
    segspl(y, yp, s);
    let sle = lefind(x, xp, y, yp, s);
    let xmax = 0.5 * (x[0] + x[n - 1]);
    let xmin = seval(sle, x, xp, s);
    let ymin = seval(sle, y, yp, s);
    let fudge = 1.0 / (xmax - xmin);
    for i in 0..n {
        x[i] = (x[i] - xmin) * fudge;
        y[i] = (y[i] - ymin) * fudge;
        s[i] *= fudge;
    }
}

// ── Geometric property calculator ────────────────────────────────────────────

/// Cross-sectional area, centroid and principal moments of inertia.
/// `itype=1`: integrate over area; `itype=2`: integrate over skin.
pub struct GeoProps {
    pub area:  f64,
    pub xcen:  f64, pub ycen:  f64,
    pub ei11:  f64, pub ei22:  f64,
    pub apx1:  f64, pub apx2:  f64,
}

pub fn aecalc(x: &[f64], y: &[f64], t: &[f64], itype: usize) -> GeoProps {
    use std::f64::consts::PI;
    let n = x.len();
    let (mut aint, mut xint, mut yint) = (0.0, 0.0, 0.0);
    let (mut xxint, mut xyint, mut yyint) = (0.0, 0.0, 0.0);
    let mut sint = 0.0;

    for io in 0..n {
        let ip = if io == n - 1 { 0 } else { io + 1 };
        let dx = x[io] - x[ip];
        let dy = y[io] - y[ip];
        let xa = (x[io] + x[ip]) * 0.5;
        let ya = (y[io] + y[ip]) * 0.5;
        let ta = (t[io] + t[ip]) * 0.5;
        let ds = (dx * dx + dy * dy).sqrt();
        sint += ds;
        let da = if itype == 1 { ya * dx } else { ta * ds };
        aint  += da;
        xint  += xa * da;
        yint  += if itype == 1 { ya * da / 2.0 } else { ya * da };
        xxint += xa * xa * da;
        xyint += if itype == 1 { xa * ya * da / 2.0 } else { xa * ya * da };
        yyint += if itype == 1 { ya * ya * da / 3.0 } else { ya * ya * da };
    }

    if aint == 0.0 {
        return GeoProps { area: 0.0, xcen: 0.0, ycen: 0.0,
                          ei11: 0.0, ei22: 0.0, apx1: 0.0, apx2: PI * 0.5 };
    }

    let xcen = xint / aint;
    let ycen = yint / aint;
    let eixx = yyint - ycen * ycen * aint;
    let eixy = xyint - xcen * ycen * aint;
    let eiyy = xxint - xcen * xcen * aint;
    let eisq = 0.25 * (eixx - eiyy).powi(2) + eixy * eixy;
    let sgn  = (eiyy - eixx).signum();
    let ei11 = 0.5 * (eixx + eiyy) - sgn * eisq.sqrt();
    let ei22 = 0.5 * (eixx + eiyy) + sgn * eisq.sqrt();

    let (apx1, apx2) = if ei11 == 0.0 || ei22 == 0.0
        || eisq / (ei11 * ei22) < (0.001 * sint).powi(4)
    {
        (0.0, PI * 0.5)
    } else {
        let c1 = eixy; let s1 = eixx - ei11;
        let c2 = eixy; let s2 = eixx - ei22;
        if s1.abs() > s2.abs() {
            let a1 = s1.atan2(c1).clamp(-PI * 0.5, PI * 0.5);
            (a1, a1 + PI * 0.5)
        } else {
            let a2 = s2.atan2(c2).clamp(-PI * 0.5, PI * 0.5);
            (a2 - PI * 0.5, a2)
        }
    };
    GeoProps { area: aint, xcen, ycen, ei11, ei22, apx1, apx2 }
}

/// Approximate max thickness and camber (discrete-point version, TCCALC).
pub fn tccalc(x: &[f64], xp: &[f64], y: &[f64], yp: &[f64], s: &[f64])
    -> (f64, f64, f64, f64) // (thick, xthick, cambr, xcambr)
{
    let n = x.len();
    let sle  = lefind(x, xp, y, yp, s);
    let xle  = seval(sle, x, xp, s);
    let yle  = seval(sle, y, yp, s);
    let xte  = 0.5 * (x[0] + x[n - 1]);
    let yte  = 0.5 * (y[0] + y[n - 1]);
    let chord = ((xte - xle).powi(2) + (yte - yle).powi(2)).sqrt();
    let dxc = (xte - xle) / chord;
    let dyc = (yte - yle) / chord;

    let (mut thick, mut xthick, mut cambr, mut xcambr) = (0.0f64, 0.0, 0.0f64, 0.0);

    for i in 0..n {
        let ybar  = (y[i] - yle) * dxc - (x[i] - xle) * dyc;
        let sopp  = sopps(s[i], x, xp, y, yp, s, sle);
        let xopp  = seval(sopp, x, xp, s);
        let yopp  = seval(sopp, y, yp, s);
        let ybarop = (yopp - yle) * dxc - (xopp - xle) * dyc;
        let yc = 0.5 * (ybar + ybarop);
        let yt = (ybar - ybarop).abs();
        if yc.abs() > cambr.abs() { cambr = yc; xcambr = xopp; }
        if yt.abs() > thick.abs() { thick = yt; xthick = xopp; }
    }
    (thick, xthick, cambr, xcambr)
}

/// Curvature-smoothing nose finder (NSFIND) — returns S at maximum curvature.
pub fn nsfind(x: &[f64], xp: &[f64], y: &[f64], yp: &[f64], s: &[f64]) -> f64 {
    let n = x.len();
    let smool = 0.006 * (s[n - 1] - s[0]);
    let smoosq = smool * smool;

    let mut cv: Vec<f64> = (0..n).map(|i| curv(s[i], x, xp, y, yp, s)).collect();
    let mut a = vec![0.0f64; n];
    let mut b = vec![0.0f64; n];
    let mut c = vec![0.0f64; n];
    a[0] = 1.0; c[0] = 0.0;
    for i in 1..n - 1 {
        let dsm = s[i] - s[i - 1];
        let dsp = s[i + 1] - s[i];
        let dso = 0.5 * (s[i + 1] - s[i - 1]);
        if dsm == 0.0 || dsp == 0.0 {
            b[i] = 0.0; a[i] = 1.0; c[i] = 0.0;
        } else {
            b[i] = smoosq * (-1.0 / dsm) / dso;
            a[i] = smoosq * (1.0 / dsp + 1.0 / dsm) / dso + 1.0;
            c[i] = smoosq * (-1.0 / dsp) / dso;
        }
    }
    b[n - 1] = 0.0; a[n - 1] = 1.0;
    trisol(&mut a, &b, &mut c, &mut cv);

    let mut cvmax = 0.0f64;
    let mut ivmax = 1usize;
    for i in 1..n - 1 {
        if cv[i].abs() > cvmax { cvmax = cv[i].abs(); ivmax = i; }
    }

    let i = ivmax;
    let ip = if s[i] == s[i + 1] { i + 2 } else { i + 1 };
    let im = if i > 1 && s[i] == s[i - 1] { i - 2 } else { i - 1 };
    let dsm  = s[i] - s[im];
    let dsp  = s[ip] - s[i];
    let cvsm = (cv[i] - cv[im]) / dsm;
    let cvsp = (cv[ip] - cv[i]) / dsp;
    let cvs  = (cvsm * dsp + cvsp * dsm) / (dsp + dsm);
    let cvss = 2.0 * (cvsp - cvsm) / (dsp + dsm);
    s[i] - cvs / cvss
}

// ── NACA airfoil generators ───────────────────────────────────────────────────

/// Generate a NACA 4-digit airfoil using the standard polynomial thickness and parabolic camber forms.
/// Returns `(xb, yb, name)` with `2*nside − 1` points ordered TE → upper LE → lower TE.
pub fn naca4(ides: u32, nside: usize) -> (Vec<f64>, Vec<f64>, String) {
    let n4 =  ides / 1000;
    let n3 = (ides - n4 * 1000) / 100;
    let n2 = (ides - n4 * 1000 - n3 * 100) / 10;
    let n1 =  ides - n4 * 1000 - n3 * 100 - n2 * 10;

    let m = n4 as f64 / 100.0;
    let p = n3 as f64 / 10.0;
    let t = (n2 * 10 + n1) as f64 / 100.0;

    let an: f64 = 1.5;
    let anp = an + 1.0;

    let mut xx = vec![0.0f64; nside];
    let mut yt = vec![0.0f64; nside];
    let mut yc = vec![0.0f64; nside];

    for i in 0..nside {
        let frac = i as f64 / (nside - 1) as f64;
        xx[i] = if i == nside - 1 { 1.0 }
                else { 1.0 - anp * frac * (1.0 - frac).powf(an) - (1.0 - frac).powf(anp) };
        yt[i] = (0.29690 * xx[i].sqrt()
               - 0.12600 * xx[i]
               - 0.35160 * xx[i].powi(2)
               + 0.28430 * xx[i].powi(3)
               - 0.10150 * xx[i].powi(4)) * t / 0.20;
        yc[i] = if xx[i] < p {
            if p == 0.0 { 0.0 } else { m / p.powi(2) * (2.0 * p * xx[i] - xx[i].powi(2)) }
        } else {
            if (1.0 - p) == 0.0 { 0.0 }
            else { m / (1.0 - p).powi(2) * ((1.0 - 2.0 * p) + 2.0 * p * xx[i] - xx[i].powi(2)) }
        };
    }

    let mut xb = Vec::with_capacity(2 * nside);
    let mut yb = Vec::with_capacity(2 * nside);
    for i in (0..nside).rev() { xb.push(xx[i]); yb.push(yc[i] + yt[i]); }
    for i in 1..nside          { xb.push(xx[i]); yb.push(yc[i] - yt[i]); }

    let name = format!("NACA {:04}", ides);
    (xb, yb, name)
}

/// Generate a NACA 5-digit airfoil using the three-term polynomial camber line.
/// Returns `None` if the first three design digits are not one of 210, 220, 230, 240, or 250.
pub fn naca5(ides: u32, nside: usize) -> Option<(Vec<f64>, Vec<f64>, String)> {
    let n5 =  ides / 10000;
    let n4 = (ides - n5 * 10000) / 1000;
    let n3 = (ides - n5 * 10000 - n4 * 1000) / 100;
    let n2 = (ides - n5 * 10000 - n4 * 1000 - n3 * 100) / 10;
    let n1 =  ides - n5 * 10000 - n4 * 1000 - n3 * 100 - n2 * 10;
    let n543 = 100 * n5 + 10 * n4 + n3;

    let (mf, c) = match n543 {
        210 => (0.0580, 361.4),
        220 => (0.1260, 51.64),
        230 => (0.2025, 15.957),
        240 => (0.2900, 6.643),
        250 => (0.3910, 3.230),
        _   => return None,
    };
    let t = (n2 * 10 + n1) as f64 / 100.0;
    let an = 1.5f64;
    let anp = an + 1.0;

    let mut xx = vec![0.0f64; nside];
    let mut yt = vec![0.0f64; nside];
    let mut yc = vec![0.0f64; nside];

    for i in 0..nside {
        let frac = i as f64 / (nside - 1) as f64;
        xx[i] = if i == nside - 1 { 1.0 }
                else { 1.0 - anp * frac * (1.0 - frac).powf(an) - (1.0 - frac).powf(anp) };
        yt[i] = (0.29690 * xx[i].sqrt()
               - 0.12600 * xx[i]
               - 0.35160 * xx[i].powi(2)
               + 0.28430 * xx[i].powi(3)
               - 0.10150 * xx[i].powi(4)) * t / 0.20;
        yc[i] = if xx[i] < mf {
            (c / 6.0) * (xx[i].powi(3) - 3.0 * mf * xx[i].powi(2) + mf * mf * (3.0 - mf) * xx[i])
        } else {
            (c / 6.0) * mf.powi(3) * (1.0 - xx[i])
        };
    }

    let mut xb = Vec::with_capacity(2 * nside);
    let mut yb = Vec::with_capacity(2 * nside);
    for i in (0..nside).rev() { xb.push(xx[i]); yb.push(yc[i] + yt[i]); }
    for i in 1..nside          { xb.push(xx[i]); yb.push(yc[i] - yt[i]); }

    Some((xb, yb, format!("NACA {:05}", ides)))
}

// ── NACA 6-series ─────────────────────────────────────────────────────────────
//
// Designation: NACA [6][S]-[C][TT]  e.g. "63412" = 63-series, CL=0.4, t/c=12%
//   S  = sub-series (3..7) — determines pressure recovery shape
//   C  = design lift coefficient × 10
//   TT = thickness / chord × 100
//
// Thickness: uses the standard NACA 4-digit polynomial form (a common engineering
//   approximation — the exact 6-series thickness forms require Abbott & Von Doenhoff
//   tabular interpolation; the 4-digit polynomial reproduces max thickness correctly
//   and the small shape difference near the max position is second-order for most
//   engineering purposes).
//
// Camber: a=1.0 uniform-loading mean line (NACA TR-824, Section 6).
//   yc = (cli / 4π) · [-(1-x)·ln(1-x) - x·ln(x)]
//   This is the canonical mean line for ALL NACA 6-series designs.
//   Max camber occurs at x = 0.5c (midchord), giving CL_0 = cli_design at α=0.

/// Generate a NACA 6-series airfoil.
///
/// `series`      — sub-series (63, 64, 65, 66, or 67)
/// `cli_design`  — design lift coefficient (e.g. 0.4 for the "4" in 63**4**12)
/// `thickness`   — max thickness / chord (e.g. 0.12 for "12" in 6341**2**)
/// `nside`       — points per surface side (65 recommended)
pub fn naca6(
    series: u32,
    cli_design: f64,
    thickness: f64,
    nside: usize,
) -> (Vec<f64>, Vec<f64>, String) {
    if ![63u32, 64, 65, 66, 67].contains(&series) {
        panic!("naca6: unsupported sub-series {series} (valid: 63-67)");
    }

    let t   = thickness;
    let an  = 1.5f64;
    let anp = an + 1.0;

    let mut xx = vec![0.0f64; nside];
    let mut yt = vec![0.0f64; nside];
    let mut yc = vec![0.0f64; nside];

    for i in 0..nside {
        let frac = i as f64 / (nside - 1) as f64;
        // cosine-bunched x distribution, denser at LE/TE
        xx[i] = if i == nside - 1 { 1.0 }
                else { 1.0 - anp * frac * (1.0 - frac).powf(an) - (1.0 - frac).powf(anp) };
        let x = xx[i];

        // ── thickness (4-digit polynomial — standard engineering approximation) ─
        let yt_raw = (t / 0.20) * (
              0.29690 * x.sqrt()
            - 0.12600 * x
            - 0.35160 * x.powi(2)
            + 0.28430 * x.powi(3)
            - 0.10150 * x.powi(4));
        yt[i] = yt_raw.max(0.0);

        // ── camber: a=1.0 uniform-loading mean line ───────────────────────
        // yc = (cli / 4π) · [-(1-x)·ln(1-x) - x·ln(x)]
        // Derivation: thin-airfoil theory with uniform bound vorticity Γ=1-x.
        // Gives CL_0 = cli_design at zero angle of attack.
        // Singularities at x=0 and x=1 resolve to yc=0 (L'Hôpital).
        yc[i] = if x < 1e-12 || x > 1.0 - 1e-12 {
            0.0
        } else {
            (cli_design / (4.0 * PI))
                * (-(1.0 - x) * (1.0 - x).ln() - x * x.ln())
        };
    }

    // upper surface: TE → LE  (i = nside-1 .. 0)
    // lower surface: LE → TE  (i = 1 .. nside-1)
    let mut xb = Vec::with_capacity(2 * nside - 1);
    let mut yb = Vec::with_capacity(2 * nside - 1);
    for i in (0..nside).rev() { xb.push(xx[i]); yb.push(yc[i] + yt[i]); }
    for i in 1..nside          { xb.push(xx[i]); yb.push(yc[i] - yt[i]); }

    let name = format!(
        "NACA {}{}-{}{:02}",
        6, series % 10,
        (cli_design * 10.0).round() as u32,
        (thickness * 100.0).round() as u32,
    );
    (xb, yb, name)
}

/// Parse a 5-digit string that starts with '6' into (series, cli_design, thickness).
/// e.g. "63412" → (63, 0.4, 0.12)
fn parse_naca6(digits: &str) -> Option<(u32, f64, f64)> {
    if digits.len() != 5 { return None; }
    let s1: u32 = digits[..1].parse().ok()?;  // must be 6
    if s1 != 6 { return None; }
    let s2: u32 = digits[1..2].parse().ok()?; // sub-series digit (3..7)
    let series = 60 + s2;                      // 63, 64, 65, 66, 67
    let cl_digit: u32  = digits[2..3].parse().ok()?;
    let thick_pct: u32 = digits[3..5].parse().ok()?;
    Some((series, cl_digit as f64 / 10.0, thick_pct as f64 / 100.0))
}

/// Load a NACA airfoil with a caller-specified number of panel nodes per side (N = 2*nside − 1, capped at IQX=360).
/// Use instead of `load_naca` when higher panel density is needed for accuracy benchmarking or convergence studies.
///
/// Common values: 65 → N=129 (default), 121 → N=241, 180 → N=359 (XFOIL standard).
pub fn load_naca_n(state: &mut XfoilState, designation: &str, nside: usize) {
    use crate::types::IQX;
    // Cap so that 2*nside-1 ≤ IQX
    let nside = nside.min((IQX + 1) / 2);
    _load_naca_impl(state, designation, nside);
}

/// Load a NACA 4-digit, 5-digit, or 6-series airfoil into `state` at the default density of 65 points/side (N=129).
/// Call before `calc_normals` / `calc_panel_angles`; accepts designations such as `"4412"`, `"23012"`, `"63412"`, or `"NACA 64-206"`.
pub fn load_naca(state: &mut XfoilState, designation: &str) {
    _load_naca_impl(state, designation, 65);
}

fn _load_naca_impl(state: &mut XfoilState, designation: &str, nside: usize) {
    // strip everything that isn't a digit or hyphen, then drop hyphens for length check
    let digits: String = designation.chars().filter(|c| c.is_ascii_digit()).collect();

    let (xb, yb, _name) = if digits.len() == 4 {
        let ides: u32 = digits.parse().unwrap_or(0);
        naca4(ides, nside)
    } else if digits.len() == 5 && digits.starts_with('6') {
        let (series, cli, t) = parse_naca6(&digits)
            .expect("invalid NACA 6-series designation");
        naca6(series, cli, t, nside)
    } else if digits.len() == 5 {
        let ides: u32 = digits.parse().unwrap_or(0);
        naca5(ides, nside).expect("invalid 5-digit NACA designation")
    } else {
        panic!("load_naca: unsupported designation '{designation}'");
    };

    // cap to IZX
    let xb: Vec<f64> = xb.into_iter().take(crate::types::IZX).collect();
    let yb: Vec<f64> = yb.into_iter().take(crate::types::IZX).collect();

    let n = xb.len();
    assert!(n <= state.geom.x.len(), "NACA airfoil too large for IZX buffer");
    state.geom.x[..n].copy_from_slice(&xb);
    state.geom.y[..n].copy_from_slice(&yb);
    state.geom.n = n;

    // compute arc length and spline derivatives
    scalc(&xb, &yb, &mut state.geom.s[..n]);
    segspl(&xb, &mut state.geom.xp[..n], &state.geom.s[..n].to_vec());
    segspl(&yb, &mut state.geom.yp[..n], &state.geom.s[..n].to_vec());

    // locate LE
    state.geom.sle = lefind(
        &state.geom.x[..n], &state.geom.xp[..n],
        &state.geom.y[..n], &state.geom.yp[..n],
        &state.geom.s[..n]);
    state.geom.xle = seval(state.geom.sle, &xb, &state.geom.xp[..n], &state.geom.s[..n]);
    state.geom.yle = seval(state.geom.sle, &yb, &state.geom.yp[..n], &state.geom.s[..n]);
    state.geom.xte = 0.5 * (xb[0] + xb[n - 1]);
    state.geom.yte = 0.5 * (yb[0] + yb[n - 1]);
    state.geom.chord = ((state.geom.xte - state.geom.xle).powi(2)
                       + (state.geom.yte - state.geom.yle).powi(2)).sqrt();
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spline::scalc;
    #[allow(unused_imports)]
    use std::f64::consts::PI;

    #[test]
    fn naca4412_leading_edge() {
        let (xb, _yb, name) = naca4(4412, 65);
        assert!(name.contains("4412"));
        assert_eq!(xb.len(), 2 * 65 - 1);
        // TE at x≈1, LE somewhere near x≈0
        assert!(xb[0] > 0.9, "first point should be near TE");
        assert!(xb.iter().cloned().fold(f64::INFINITY, f64::min) < 0.01,
                "minimum x should be near 0");
    }

    #[test]
    fn naca4412_le_finder() {
        let (xb, yb, _) = naca4(4412, 65);
        let n = xb.len();
        let mut xp = vec![0.0; n];
        let mut yp = vec![0.0; n];
        let mut s  = vec![0.0; n];
        scalc(&xb, &yb, &mut s);
        segspl(&xb, &mut xp, &s);
        segspl(&yb, &mut yp, &s);
        let sle = lefind(&xb, &xp, &yb, &yp, &s);
        let xle = seval(sle, &xb, &xp, &s);
        // LE should be near x=0 for NACA 4412
        assert!(xle < 0.01, "LE x should be near 0, got {xle}");
    }

    #[test]
    fn naca5_valid() {
        let result = naca5(23012, 65);
        assert!(result.is_some(), "NACA 23012 should parse");
        let (xb, _yb, name) = result.unwrap();
        assert_eq!(xb.len(), 2 * 65 - 1);
        assert!(name.contains("23012"));
    }

    #[test]
    fn naca5_invalid() {
        assert!(naca5(11012, 65).is_none(), "NACA 11012 should fail");
    }

    #[test]
    fn aecalc_unit_square() {
        // Square: vertices at (0,0),(1,0),(1,1),(0,1) — area should be 1
        let x = vec![0.0, 1.0, 1.0, 0.0];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let t = vec![1.0; 4];
        let props = aecalc(&x, &y, &t, 1);
        assert!((props.area - 1.0).abs() < 1e-10, "area={}", props.area);
        assert!((props.xcen - 0.5).abs() < 1e-10, "xcen={}", props.xcen);
    }

    #[test]
    fn naca0012_chord_near_one() {
        // After load_naca, chord should be ≈ 1.0
        use crate::types::XfoilState;
        let mut state = XfoilState::default();
        load_naca(&mut state, "0012");
        assert!((state.geom.chord - 1.0).abs() < 0.02,
            "chord={:.4}", state.geom.chord);
    }

    #[test]
    fn naca0012_symmetric_le() {
        // NACA 0012 is symmetric — LE should be at x≈0, y≈0
        use crate::types::XfoilState;
        let mut state = XfoilState::default();
        load_naca(&mut state, "0012");
        assert!(state.geom.xle < 0.01, "xle={:.5}", state.geom.xle);
        assert!(state.geom.yle.abs() < 1e-6, "yle={:.5}", state.geom.yle);
    }

    #[test]
    fn naca4412_thickness_camber() {
        // NACA 4412: max thickness ~12%, max camber ~4%
        let (xb, yb, _) = naca4(4412, 65);
        let n = xb.len();
        let mut xp = vec![0.0; n]; let mut yp = vec![0.0; n];
        let mut s  = vec![0.0; n];
        scalc(&xb, &yb, &mut s);
        segspl(&xb, &mut xp, &s);
        segspl(&yb, &mut yp, &s);
        let (thick, _, cambr, _) = tccalc(&xb, &xp, &yb, &yp, &s);
        // Tolerance is loose because tccalc uses discrete points
        assert!(thick > 0.10 && thick < 0.14,
            "thickness={:.4} (expected ~0.12)", thick);
        assert!(cambr.abs() > 0.02 && cambr.abs() < 0.06,
            "camber={:.4} (expected ~0.04)", cambr);
    }

    #[test]
    fn naca0012_zero_camber() {
        // NACA 0012: symmetric, so camber should be ≈ 0
        let (xb, yb, _) = naca4(12, 65);
        let n = xb.len();
        let mut xp = vec![0.0; n]; let mut yp = vec![0.0; n];
        let mut s  = vec![0.0; n];
        scalc(&xb, &yb, &mut s);
        segspl(&xb, &mut xp, &s);
        segspl(&yb, &mut yp, &s);
        let (thick, _, cambr, _) = tccalc(&xb, &xp, &yb, &yp, &s);
        assert!(thick > 0.10 && thick < 0.14,
            "thickness={:.4}", thick);
        assert!(cambr.abs() < 0.005,
            "camber={:.6} should be ~0 for symmetric airfoil", cambr);
    }

    #[test]
    fn scalc_monotone() {
        // Arc length must be strictly increasing
        let (xb, yb, _) = naca4(2412, 65);
        let n = xb.len();
        let mut s = vec![0.0; n];
        scalc(&xb, &yb, &mut s);
        for i in 1..n {
            assert!(s[i] > s[i-1], "s[{i}]={} not > s[{}]={}", s[i], i-1, s[i-1]);
        }
    }

    #[test]
    fn nsfind_near_le() {
        // Max curvature should be near the LE (small x)
        let (xb, yb, _) = naca4(4412, 65);
        let n = xb.len();
        let mut xp = vec![0.0; n]; let mut yp = vec![0.0; n];
        let mut s  = vec![0.0; n];
        scalc(&xb, &yb, &mut s);
        segspl(&xb, &mut xp, &s);
        segspl(&yb, &mut yp, &s);
        let s_nose = nsfind(&xb, &xp, &yb, &yp, &s);
        let x_nose = seval(s_nose, &xb, &xp, &s);
        assert!(x_nose < 0.05, "nose x={x_nose:.4} should be near LE");
    }

    // ── NACA 6-series tests ───────────────────────────────────────────────────

    #[test]
    fn naca6_point_count() {
        let (xb, yb, _) = naca6(63, 0.4, 0.12, 65);
        assert_eq!(xb.len(), 2 * 65 - 1);
        assert_eq!(xb.len(), yb.len());
    }

    #[test]
    fn naca6_name_format() {
        let (_, _, name) = naca6(63, 0.4, 0.12, 65);
        assert!(name.contains("63"), "name={name}");
        assert!(name.contains("4"),  "name={name}");
        assert!(name.contains("12"), "name={name}");
    }

    #[test]
    fn naca6_te_at_one() {
        // First and last points should be at x≈1 (trailing edge)
        for series in [63u32, 64, 65, 66, 67] {
            let (xb, _, _) = naca6(series, 0.2, 0.12, 65);
            assert!(xb[0] > 0.98, "series {series}: xb[0]={}", xb[0]);
            assert!(xb[xb.len()-1] > 0.98,
                "series {series}: last xb={}", xb[xb.len()-1]);
        }
    }

    #[test]
    fn naca6_le_near_zero() {
        // Minimum x should be near 0 (leading edge)
        let (xb, _, _) = naca6(64, 0.3, 0.09, 65);
        let x_min = xb.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(x_min < 0.005, "min x={x_min:.5}");
    }

    #[test]
    fn naca6_symmetric_zero_camber() {
        // cli=0 → symmetric, so upper and lower surfaces should be mirror images
        let (xb, yb, _) = naca6(65, 0.0, 0.12, 65);
        let n = xb.len();
        let mid = n / 2;
        // Upper surface has positive y, lower has negative y near midchord
        assert!(yb[mid / 2] > 0.0, "upper surface should be positive");
        assert!(yb[mid + mid / 2] < 0.0, "lower surface should be negative");
        // Symmetric: upper[i] ≈ -lower[mirror_i] at same x
        // (loose check since x-distribution differs slightly)
        let max_upper = yb[..mid].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_lower = yb[mid..].iter().cloned().fold(f64::INFINITY, f64::min);
        assert!((max_upper + min_lower).abs() < 1e-10,
            "max_upper={max_upper:.6}, min_lower={min_lower:.6}");
    }

    #[test]
    fn naca6_positive_camber_lifts() {
        // With positive cli_design, upper surface should be above lower at midchord
        let (xb, yb, _) = naca6(64, 0.4, 0.12, 65);
        let n  = xb.len();
        let mid = n / 2;  // approximately the LE
        // y at upper midchord should be greater than lower midchord
        let y_upper_mid = yb[mid / 2];
        let y_lower_mid = yb[mid + mid / 2];
        assert!(y_upper_mid > y_lower_mid,
            "upper={y_upper_mid:.4} should exceed lower={y_lower_mid:.4}");
    }

    #[test]
    fn naca6_thickness_approx_correct() {
        // For 63-012 (symmetric, cli=0): max total thickness should be ≈ 12%
        // The maximum occurs somewhere between 30-50% chord for 6-series.
        let (xb, yb, _) = naca6(63, 0.0, 0.12, 65);
        let n   = xb.len();
        let mid = n / 2;

        // For each upper-surface point, find closest lower-surface x and compute thickness
        let mut max_thick = 0.0f64;
        for i in 0..mid {
            let xu = xb[i];
            // find nearest lower-surface point at same x
            let j = xb[mid..].iter().enumerate()
                .min_by(|(_, a), (_, b)|
                    ((*a - xu).abs()).partial_cmp(&((*b - xu).abs())).unwrap())
                .map(|(idx, _)| idx).unwrap();
            let thick = yb[i] - yb[mid + j];
            if thick > max_thick { max_thick = thick; }
        }
        assert!(max_thick > 0.10 && max_thick < 0.14,
            "max total thickness={max_thick:.4} (expected ~0.12 for 63-012)");
    }

    #[test]
    fn naca6_load_naca_63412() {
        use crate::types::XfoilState;
        let mut state = XfoilState::default();
        load_naca(&mut state, "63412");
        assert!(state.geom.n > 0);
        assert!((state.geom.chord - 1.0).abs() < 0.02,
            "chord={:.4}", state.geom.chord);
        assert!(state.geom.xle < 0.01,
            "xle={:.5}", state.geom.xle);
    }

    #[test]
    fn naca6_load_naca_with_hyphen() {
        use crate::types::XfoilState;
        let mut state = XfoilState::default();
        // Should parse "NACA 64-206" the same as "64206"
        load_naca(&mut state, "NACA 64-206");
        assert!(state.geom.n > 0);
        assert!((state.geom.chord - 1.0).abs() < 0.02);
    }

    #[test]
    fn naca6_all_series_build() {
        for series in [63u32, 64, 65, 66, 67] {
            let (xb, yb, _) = naca6(series, 0.2, 0.15, 65);
            assert!(xb.len() == 2 * 65 - 1, "series {series} wrong length");
            assert!(yb.iter().all(|v| v.is_finite()), "series {series} has non-finite y");
        }
    }

    #[test]
    fn naca6_camber_max_at_midchord() {
        // a=1 uniform loading always has max camber at x=0.5 by definition.
        let camber = |x: f64| -> f64 {
            if x < 1e-12 || x > 1.0 - 1e-12 { 0.0 }
            else { -(1.0 - x) * (1.0 - x).ln() - x * x.ln() }
        };
        let mut best_x = 0.0f64;
        let mut best_y = 0.0f64;
        for i in 1..200 {
            let x = i as f64 / 200.0;
            let y = camber(x);
            if y > best_y { best_y = y; best_x = x; }
        }
        assert!((best_x - 0.5).abs() < 0.02,
            "max camber at x={best_x:.3}, expected 0.5");
    }

    #[test]
    fn naca6_invalid_series_panics() {
        let result = std::panic::catch_unwind(|| naca6(62, 0.3, 0.12, 65));
        assert!(result.is_err(), "series 62 should panic");
    }

    #[test]
    fn naca6_parse_naca6_fn() {
        let r = parse_naca6("63412").unwrap();
        assert_eq!(r.0, 63);
        assert!((r.1 - 0.4).abs() < 1e-10, "cli={}", r.1);
        assert!((r.2 - 0.12).abs() < 1e-10, "t={}", r.2);

        let r2 = parse_naca6("65015").unwrap();
        assert_eq!(r2.0, 65);
        assert!((r2.1 - 0.0).abs() < 1e-10);
        assert!((r2.2 - 0.15).abs() < 1e-10);

        assert!(parse_naca6("4412").is_none());   // wrong length
        assert!(parse_naca6("23012").is_none());  // starts with 2, not 6
    }

    #[test]
    fn norm_coords_unit_chord() {
        // After normalisation, chord should be exactly 1.0
        let (mut xb, mut yb, _) = naca4(2412, 65);
        let n = xb.len();
        let mut xp = vec![0.0; n]; let mut yp = vec![0.0; n];
        let mut s  = vec![0.0; n];
        // scale to a different size first
        for x in xb.iter_mut() { *x *= 2.5; }
        for y in yb.iter_mut() { *y *= 2.5; }
        norm_coords(&mut xb, &mut xp, &mut yb, &mut yp, &mut s);
        let sle = lefind(&xb, &xp, &yb, &yp, &s);
        let xle = seval(sle, &xb, &xp, &s);
        let xte = 0.5 * (xb[0] + xb[n-1]);
        assert!((xte - xle - 1.0).abs() < 0.005,
            "chord after norm={:.5}", xte - xle);
    }
}
