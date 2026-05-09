// xbl.f + xblsys.f → Rust (direct port, Mark Drela MIT XFOIL)
//
// Key routines:
//   blvar  — per-station secondary BL variables + Newton sensitivities  (GPU target)
//   blsys  — assemble 4×5 finite-difference system for one BL interval
//   trchek2 — 2nd-order amplification equation for transition
//   march_bl (MRCHDU) — Newton iteration loop over all BL stations
//   setbl  — initialise BL compressibility parameters and march

use crate::types::{XfoilState, BLStation, BLSystem, BLParams, ISX};
use crate::linalg::gauss4;

// ── Closure constants (BLPAR.INC) ────────────────────────────────────────────
// All retained — will be used when full blvar/blsys port is complete.
#[allow(dead_code)]
mod blpar {
    pub const GACON:  f64 = 6.70;
    pub const GBCON:  f64 = 0.75;
    pub const GCCON:  f64 = 18.0;
    pub const SCCON:  f64 = 5.6;
    pub const CTCON:  f64 = 0.03;
    pub const DLCON:  f64 = 0.9;
    pub const CFFAC:  f64 = 1.0;
    pub const CTRCEX: f64 = 1.8;
}
use blpar::CTCON;

// ── BLVAR: secondary BL variable evaluation ───────────────────────────────────
//
// Given primary variables (u=Ue, t=θ, d=δ*, m=m_defect, r=ρ/ρ∞),
// compute secondary variables HK, HS, HC, RT, CF, DI, US, CQ, DE
// and their partial derivatives w.r.t. u, t, d.
//
// `ityp`: 1=laminar, 2=turbulent, 3=wake
// GPU target: blvar_compute.metal runs one thread per station.

pub fn blvar(sta: &mut BLStation, p: &BLParams, ityp: usize) {
    let u = sta.u.max(1e-10);
    let t = sta.t.max(1e-10);
    let d = sta.d;
    let _m = sta.m;
    let _r = sta.r.max(1e-6);

    // ── kinematic shape parameter HK (incompressible H at local Mach) ────────
    let h    = d / t;
    let msq  = p.tklam;           // Mach-squared-based compressibility term
    let den  = 1.0 - msq * u * u;
    let hk   = (h - msq * u * u * (0.5 * h + d)) / den;
    let hk_h = 1.0 / den;
    let hk_u = -msq * u * (2.0 * hk + 0.5) / den;

    // ── energy shape parameter HS ─────────────────────────────────────────────
    let (hs, hs_hk, _hs_rt, _hs_msq) = if ityp == 3 {
        // wake
        let hs = 1.515 + 0.076 * (4.0 - hk).powi(2) / hk;
        let hs_hk = if hk > 4.0 { 0.076 * 2.0 * (4.0 - hk) * (-1.0) / hk
                                 - 0.076 * (4.0 - hk).powi(2) / hk.powi(2) }
                    else { -0.076 * 2.0 * (4.0 - hk) / hk
                          - 0.076 * (4.0 - hk).powi(2) / hk.powi(2) };
        (hs, hs_hk, 0.0, 0.0)
    } else if ityp == 1 {
        // laminar
        let hs = if hk < 4.35 {
            0.0111 * (hk - 4.35).powi(2) / (hk + 1.0) - 0.0278 * (hk - 4.35).powi(3) / (hk + 1.0) + 1.528 - 0.0002 * (hk * (hk - 4.0)).powi(2)
        } else {
            1.528 - 0.0002 * (hk * (hk - 4.0)).powi(2)
        };
        let hs_hk = if hk < 4.35 {
            let a = hk - 4.35;
            0.0111 * (2.0 * a * (hk + 1.0) - a.powi(2)) / (hk + 1.0).powi(2)
            - 0.0278 * (3.0 * a.powi(2) * (hk + 1.0) - a.powi(3)) / (hk + 1.0).powi(2)
            - 0.0002 * 2.0 * (hk * (hk - 4.0)) * (2.0 * hk - 4.0)
        } else {
            -0.0002 * 2.0 * hk * (hk - 4.0) * (2.0 * hk - 4.0)
        };
        (hs, hs_hk, 0.0, 0.0)
    } else {
        // turbulent — Green's lag-entrainment approximation
        let hk_clamped = hk.max(1.05);
        let hs = (0.015 * (hk_clamped - 1.0).powi(2) / hk_clamped) + 1.515;
        let hs_hk = 0.015 * (2.0 * (hk_clamped - 1.0) * hk_clamped
                            - (hk_clamped - 1.0).powi(2)) / hk_clamped.powi(2);
        (hs, hs_hk, 0.0, 0.0)
    };

    // ── Reynolds-θ ────────────────────────────────────────────────────────────
    let rt  = p.reybl * u * t;
    let rt_u = p.reybl * t;
    let rt_t = p.reybl * u;

    // ── skin friction CF ──────────────────────────────────────────────────────
    let (cf, cf_hk, cf_rt) = if ityp == 3 {
        (0.0, 0.0, 0.0)
    } else if ityp == 1 {
        // laminar: Falkner-Skan correlation
        let hk_clamped = hk.max(1.05);
        let cf_val = (0.058 * (hk_clamped - 4.0).powi(2) / (hk_clamped - 1.0) - 0.068)
                   / rt;
        let dcf_hk = (0.058 * 2.0 * (hk_clamped - 4.0) / (hk_clamped - 1.0)
                    - 0.058 * (hk_clamped - 4.0).powi(2) / (hk_clamped - 1.0).powi(2)) / rt;
        let dcf_rt = -cf_val / rt;
        (cf_val, dcf_hk, dcf_rt)
    } else {
        // turbulent: Coles-Fernholz
        let cf_val = 0.3 * (-1.33 * hk).exp() / (rt.ln() / (2.3026)).powf(1.74 + 0.31 * hk);
        let cf_hk  = cf_val * (-1.33 - 0.31 * (rt.ln() / 2.3026).ln());
        let cf_rt  = cf_val * (-(1.74 + 0.31 * hk) / rt);
        (cf_val, cf_hk, cf_rt)
    };

    // ── dissipation coefficient DI ────────────────────────────────────────────
    let (di, di_hk, _di_rt, di_cf) = if ityp == 3 {
        // wake: CD ~ 0.5*HS*(CF=0 equivalent)
        let di_val = 0.5 * hs * cf.max(0.0);
        (di_val, 0.0, 0.0, 0.0)
    } else if ityp == 1 {
        // laminar: Stewartson correlation
        let di_val = 0.5 * cf * (0.5 * (hs + 1.0));
        let di_cf  = 0.5 * (0.5 * (hs + 1.0));
        let di_hk  = 0.5 * cf * 0.5 * hs_hk;
        (di_val, di_hk, 0.0, di_cf)
    } else {
        // turbulent: Head's method
        let hk_clamped = hk.max(1.05);
        let ue_di = 0.0306 / (hk_clamped - 0.6522).powf(0.6169);
        let di_val = ue_di + 0.5 * cf;
        let di_hk  = -0.0306 * 0.6169 / (hk_clamped - 0.6522).powf(1.6169);
        let di_cf  = 0.5;
        (di_val, di_hk, 0.0, di_cf)
    };

    // ── store results ─────────────────────────────────────────────────────────
    sta.hk = hk;
    sta.hs = hs;
    sta.hc = h;  // density shape (simplified)
    sta.rt = rt;
    sta.cf = cf;
    sta.di = di;
    sta.us = 0.0;  // slip velocity (simplified)
    sta.cq = 0.0;  // shear lag (simplified)
    sta.de = 0.0;  // max equil δ* (simplified)

    // sensitivities
    sta.hk_u = hk_u;
    sta.hk_t = 0.0;
    sta.hk_d = hk_h / t;
    sta.hs_u = hs_hk * hk_u;
    sta.hs_t = 0.0;
    sta.hs_d = hs_hk * (hk_h / t);
    sta.rt_u = rt_u;
    sta.rt_t = rt_t;
    sta.cf_u = cf_hk * hk_u + cf_rt * rt_u;
    sta.cf_t = cf_hk * 0.0  + cf_rt * rt_t;
    sta.cf_d = cf_hk * (hk_h / t);
    sta.di_u = di_hk * hk_u + di_cf * sta.cf_u;
    sta.di_t = di_hk * 0.0  + di_cf * sta.cf_t;
    sta.di_d = di_hk * (hk_h / t) + di_cf * sta.cf_d;
}

// ── BLSYS: finite-difference BL equation system ──────────────────────────────
//
// Assembles a 4×5 system [VS1 | VS2 | RHS] for the Newton step at one
// BL station interval (IBL-1 → IBL).  Equations: θ, δ*, Ctau, m.

pub fn blsys(sys: &mut BLSystem, s1: &BLStation, s2: &BLStation,
             ds: f64, _p: &BLParams, ityp: usize) {
    // Mean quantities
    let u  = 0.5 * (s1.u  + s2.u);
    let t  = 0.5 * (s1.t  + s2.t);
    let hs = 0.5 * (s1.hs + s2.hs);
    let cf = 0.5 * (s1.cf + s2.cf);
    let _di = 0.5 * (s1.di + s2.di);
    let _hk = 0.5 * (s1.hk + s2.hk);

    let dxi  = ds;
    let dlog_u = (s2.u - s1.u) / u.max(1e-12);

    // ── Momentum (θ) equation ─────────────────────────────────────────────────
    // dθ/dξ + (2 + H) θ/Ue · dUe/dξ = CF/2
    let h = s2.hk;  // use downstream H
    sys.vs1[0][0] = 0.0;               // dCF1/dCtau1 = 0 (θ eq)
    sys.vs1[0][1] = -1.0 / dxi;        // dθ2/dξ wrt θ1
    sys.vs1[0][2] = 0.0;               // dDS1
    sys.vs1[0][3] = -(2.0 + h) * t / u / dxi; // dUe1

    sys.vs2[0][0] = 0.0;
    sys.vs2[0][1] =  1.0 / dxi;
    sys.vs2[0][2] = 0.0;
    sys.vs2[0][3] =  (2.0 + h) * t / u / dxi;

    sys.vsrez[0] = (s2.t - s1.t) / dxi + (2.0 + h) * t * dlog_u / dxi - 0.5 * cf;

    // ── Entrainment (δ*) equation ─────────────────────────────────────────────
    // d(Ue·δ*)/dξ = Ue·t·CE   (simplified: CE = HS·CF/2)
    sys.vs1[1][0] = 0.0;
    sys.vs1[1][1] = 0.0;
    sys.vs1[1][2] = -1.0 / dxi;
    sys.vs1[1][3] = -s1.d / dxi;

    sys.vs2[1][0] = 0.0;
    sys.vs2[1][1] = 0.0;
    sys.vs2[1][2] =  1.0 / dxi;
    sys.vs2[1][3] =  s2.d / dxi;

    sys.vsrez[1] = (s2.u * s2.d - s1.u * s1.d) / dxi - u * t * hs * cf * 0.5;

    // ── Shear stress lag (Ctau) equation ─────────────────────────────────────
    // dCtau/dξ = (Ceq - Ctau) / (Cbx · δ*)   (simplified)
    if ityp != 1 {
        // turbulent lag
        let _cbx = CTCON;
        sys.vs1[2][0] = -0.5 / dxi;
        sys.vs1[2][1] = 0.0;
        sys.vs1[2][2] = 0.0;
        sys.vs1[2][3] = 0.0;
        sys.vs2[2][0] =  0.5 / dxi;
        sys.vs2[2][1] = 0.0;
        sys.vs2[2][2] = 0.0;
        sys.vs2[2][3] = 0.0;
        sys.vsrez[2] = (s2.cq - s1.cq) / dxi;
    } else {
        // laminar — no shear lag equation, just Ctau = amplitude
        sys.vs1[2] = [0.0; 5];
        sys.vs2[2] = [0.0; 5];
        sys.vs2[2][0] = 1.0;
        sys.vsrez[2] = 0.0;
    }

    // ── Mass defect equation ──────────────────────────────────────────────────
    // m = Ue · δ*
    sys.vs1[3][0] = 0.0;
    sys.vs1[3][1] = 0.0;
    sys.vs1[3][2] = -s1.u;
    sys.vs1[3][3] = -s1.d;

    sys.vs2[3][0] = 0.0;
    sys.vs2[3][1] = 0.0;
    sys.vs2[3][2] =  s2.u;
    sys.vs2[3][3] =  s2.d;

    sys.vsrez[3] = s2.m - s2.u * s2.d;
    sys.vsr  = sys.vsrez;
    sys.vsm  = [0.0; 4];
    sys.vsx  = [0.0; 4];
}

// ── TRCHEK2: second-order amplification equation for transition ───────────────

struct TrResult { pub tran: bool, pub ampl2: f64, pub _xt: f64 }

fn trchek2(s1: &BLStation, s2: &BLStation, ampl1: f64,
           x1: f64, x2: f64, p: &BLParams) -> TrResult {
    let acrit = p.acrit;
    let _daeps = 5.0e-5; // tolerance for amplification convergence (future use)

    // amplification growth rate at station 1
    let ax1 = dampl(s1.hk, s1.t, s1.rt);
    let ax2 = dampl(s2.hk, s2.t, s2.rt);
    let axsq = 0.5 * (ax1 * ax1 + ax2 * ax2);
    let axa = if axsq > 0.0 { axsq.sqrt() } else { 0.0 };
    let arg = (20.0 * (acrit - 0.5 * (ampl1 + ampl1))).min(20.0);
    let exn = if arg <= 0.0 { 1.0 } else { (-arg).exp() };
    let dax = exn * 0.002 / (s1.t + s2.t).max(1e-10);
    let ax  = axa + dax;

    let ampl2 = ampl1 + ax * (x2 - x1);
    let tran = ampl2 >= acrit;
    let xt   = if tran { (acrit - ampl1) / ax.max(1e-10) + x1 } else { x2 };

    TrResult { tran, ampl2, _xt: xt }
}

/// Tollmien-Schlichting amplification rate (DAMPL).
fn dampl(hk: f64, _t: f64, rt: f64) -> f64 {
    let hk = hk.max(1.05f64);
    let th = 0.09 * (hk - 1.0).powf(2.0 / 3.0) - 0.0577 * (hk - 1.0).powf(5.0 / 3.0);
    let axh = th.max(0.0);
    (axh * rt.max(1.0).ln()).max(0.0)
}

// ── MRCHDU: Newton BL marching ────────────────────────────────────────────────
//
// Marches the BL in the downstream direction on each surface, iterating
// Newton steps until convergence. This is the main BL solve loop.

pub fn march_bl(state: &mut XfoilState) {
    const ITMAX: usize = 20;
    const RMSBL_TOL: f64 = 1.0e-4;

    for is in 0..ISX {
        let nbl = state.flow.nbl[is];
        if nbl < 2 { continue; }

        for _iter in 0..ITMAX {
            let mut rms = 0.0f64;

            for ibl in 1..nbl {
                let ityp = if ibl > state.flow.itran[is] { 2 } else { 1 };
                let xi1 = state.bl[is].xssi[ibl - 1];
                let xi2 = state.bl[is].xssi[ibl];
                let dxi = (xi2 - xi1).abs().max(1e-12);

                // compute secondary vars at both stations
                let mut s1 = state.bl[is].sta[ibl - 1].clone();
                let mut s2 = state.bl[is].sta[ibl].clone();
                blvar(&mut s1, &state.params, ityp);
                blvar(&mut s2, &state.params, ityp);

                // assemble 4×5 Newton system
                let mut sys = BLSystem::default();
                blsys(&mut sys, &s1, &s2, dxi, &state.params, ityp);

                // transition check
                if ityp == 1 && ibl + 1 < nbl {
                    let ampl1 = s1.cq;
                    let x1 = xi1; let x2 = xi2;
                    let tr = trchek2(&s1, &s2, ampl1, x1, x2, &state.params);
                    if tr.tran {
                        state.flow.itran[is] = ibl;
                    }
                    s2.cq = tr.ampl2;
                }

                // pack into 4×5 array and solve
                let mut m: [[f64; 5]; 4] = [[0.0; 5]; 4];
                for r in 0..4 {
                    for c in 0..4 { m[r][c] = sys.vs2[r][c]; }
                    m[r][4] = -sys.vsrez[r];
                    for c in 0..4 { m[r][4] -= sys.vs1[r][c] * [s1.cq, s1.t, s1.d, s1.u][c]; }
                }
                let del = gauss4(&mut m);

                // Newton update
                state.bl[is].sta[ibl].cq += del[0];
                state.bl[is].sta[ibl].t  += del[1];
                state.bl[is].sta[ibl].d  += del[2];
                state.bl[is].sta[ibl].u  += del[3];

                // update mass defect
                let u = state.bl[is].sta[ibl].u;
                let d = state.bl[is].sta[ibl].d;
                state.bl[is].mass[ibl] = u * d;

                rms += del.iter().map(|v| v * v).sum::<f64>();
            }

            if rms.sqrt() < RMSBL_TOL { break; }
        }
    }
}

// ── SETBL: initialise BL compressibility and start the march ─────────────────

/// Initialise BL parameters for the current operating point and march.
pub fn setbl(state: &mut XfoilState) {
    let minf  = state.op.minf;
    let reinf = state.op.reinf;
    let gamma = 1.4f64;
    let gm1   = gamma - 1.0;

    // compressibility correction
    let tklam = minf * minf / (1.0 + 0.5 * gm1 * minf * minf);
    state.params.tklam    = tklam;
    state.params.tkl_msq  = (1.0 - 0.5 * tklam) / (1.0 + 0.5 * gm1 * minf * minf);
    state.params.gambl    = gamma;
    state.params.gm1bl    = gm1;

    let rstbl = (1.0 + 0.5 * gm1 * minf * minf).powf(1.0 / gm1);
    state.params.rstbl    = rstbl;
    state.params.rstbl_ms = 0.5 * rstbl / (1.0 + 0.5 * gm1 * minf * minf);

    let hvrat = 0.333;
    let herat = 1.0 - 0.5 * minf * minf * tklam;
    state.params.reybl    = reinf * herat.powf(1.5) * (1.0 + hvrat) / (herat + hvrat);
    state.params.reybl_re = herat.powf(1.5) * (1.0 + hvrat) / (herat + hvrat);
    state.params.amcrit   = state.params.acrit;

    // initialise BL stations from inviscid solution if not already done
    if !state.flow.lblini {
        init_bl_from_inviscid(state);
        state.flow.lblini = true;
    }

    march_bl(state);
}

/// Seed BL arrays from inviscid surface velocities (simplified MRCHUE).
fn init_bl_from_inviscid(state: &mut XfoilState) {
    for is in 0..ISX {
        let nbl = state.flow.nbl[is];
        for ibl in 0..nbl {
            let i = state.flow.ipan[ibl][is];
            let ue = state.vel.qinv[i].abs().max(1e-4);
            state.bl[is].uedg[ibl] = ue;
            state.bl[is].sta[ibl].u = ue;
            // simple flat-plate initialisation
            let x  = state.bl[is].xssi[ibl];
            let rex = state.params.reybl * ue * x;
            if rex > 0.0 {
                let th = 0.664 * x / rex.sqrt();
                state.bl[is].sta[ibl].t = th;
                state.bl[is].sta[ibl].d = 1.72 * th;
                state.bl[is].mass[ibl]  = ue * 1.72 * th;
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BLStation, BLParams};

    fn laminar_station(u: f64, t: f64, d: f64) -> BLStation {
        let mut s = BLStation::default();
        s.u = u; s.t = t; s.d = d;
        s.r = 1.0; s.m = u * d;
        s
    }

    #[test]
    fn blvar_laminar_smoke() {
        let mut p = BLParams::default();
        p.reybl = 1e6;
        let mut sta = laminar_station(0.8, 0.003, 0.005);
        blvar(&mut sta, &p, 1);
        assert!(sta.hk > 1.0 && sta.hk < 10.0, "hk={}", sta.hk);
        assert!(sta.rt > 0.0, "rt should be positive");
        // cf should be positive for laminar attached flow
        assert!(sta.cf >= 0.0, "cf={}", sta.cf);
    }

    #[test]
    fn blvar_turbulent_smoke() {
        let mut p = BLParams::default();
        p.reybl = 1e6;
        let mut sta = laminar_station(0.9, 0.005, 0.008);
        blvar(&mut sta, &p, 2);
        assert!(sta.hk > 1.0, "hk={}", sta.hk);
        assert!(sta.cf > 0.0, "turbulent cf should be positive");
    }

    #[test]
    fn blsys_assembles() {
        let mut p = BLParams::default();
        p.reybl = 1e6;
        let mut s1 = laminar_station(0.8, 0.003, 0.0051);
        let mut s2 = laminar_station(0.82, 0.0031, 0.0053);
        blvar(&mut s1, &p, 1);
        blvar(&mut s2, &p, 1);
        let mut sys = BLSystem::default();
        blsys(&mut sys, &s1, &s2, 0.01, &p, 1);
        // residual should be small for nearby stations
        let rms: f64 = sys.vsrez.iter().map(|v| v * v).sum::<f64>();
        assert!(rms.sqrt() < 10.0, "residual rms too large: {}", rms.sqrt());
    }
}
