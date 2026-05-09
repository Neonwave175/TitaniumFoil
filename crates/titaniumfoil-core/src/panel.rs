// xpanel.f → Rust (direct port, Mark Drela MIT XFOIL)
//
// GPU target: the inner (i,j) loop of psilin / ggcalc.
// CPU version here is for correctness validation.

use std::f64::consts::PI;
use crate::types::XfoilState;
use crate::spline::segspl;
use crate::linalg::{ludcmp, baksub};

// Fortran XFOIL: QOPI = 0.25/PI = 1/(4π), HOPI = 0.50/PI = 1/(2π).
// "Quarter Over PI" and "Half Over PI" — counterintuitive names.
const QOPI: f64 = 0.25 / PI; // 1/(4π) — panel vortex/source kernels
const HOPI: f64 = 0.50 / PI; // 1/(2π) — TE half-panel contribution

/// Sentinel: pass as `io` when evaluating at an off-body point (not an airfoil/wake node).
/// Fortran uses IO=0 for this; Rust is 0-indexed so we use usize::MAX to avoid
/// collision with airfoil node 0.
pub const PSILIN_OFFBODY: usize = usize::MAX;

// ── Normal vectors and panel angles ──────────────────────────────────────────

/// Compute outward unit normals and spline derivatives for all panel nodes.
/// Call once after geometry is loaded and before any panel or boundary-layer computation.
pub fn calc_normals(state: &mut XfoilState) {
    let n = state.geom.n;
    let s = state.geom.s[..n].to_vec();
    let x = state.geom.x[..n].to_vec();
    let y = state.geom.y[..n].to_vec();

    segspl(&x, &mut state.geom.xp[..n], &s);
    segspl(&y, &mut state.geom.yp[..n], &s);

    for i in 0..n {
        let sx =  state.geom.yp[i];
        let sy = -state.geom.xp[i];
        let smod = (sx * sx + sy * sy).sqrt();
        state.geom.nx[i] = sx / smod;
        state.geom.ny[i] = sy / smod;
    }

    // average normals at corner (duplicate S) points
    for i in 0..n - 1 {
        if s[i] == s[i + 1] {
            let sx = 0.5 * (state.geom.nx[i] + state.geom.nx[i + 1]);
            let sy = 0.5 * (state.geom.ny[i] + state.geom.ny[i + 1]);
            let smod = (sx * sx + sy * sy).sqrt();
            state.geom.nx[i]     = sx / smod;
            state.geom.ny[i]     = sy / smod;
            state.geom.nx[i + 1] = sx / smod;
            state.geom.ny[i + 1] = sy / smod;
        }
    }
}

/// Compute the panel orientation angle (atan2 of panel tangent) for each panel.
/// Call after `calc_normals`; required before PSILIN or the GPU kernel.
pub fn calc_panel_angles(state: &mut XfoilState) {
    let n = state.geom.n;
    for i in 0..n - 1 {
        let sx = state.geom.x[i + 1] - state.geom.x[i];
        let sy = state.geom.y[i + 1] - state.geom.y[i];
        state.geom.apanel[i] = if sx == 0.0 && sy == 0.0 {
            (-state.geom.ny[i]).atan2(-state.geom.nx[i])
        } else {
            sx.atan2(-sy)
        };
    }
    if state.flow.sharp {
        state.geom.apanel[n - 1] = PI;
    } else {
        let sx = state.geom.x[0] - state.geom.x[n - 1];
        let sy = state.geom.y[0] - state.geom.y[n - 1];
        state.geom.apanel[n - 1] = (-sx).atan2(sy) + PI;
    }
}

// ── PSILIN ────────────────────────────────────────────────────────────────────
//
// Computes streamfunction Ψ and dΨ/dn at point (xi,yi) due to:
//   - bound vorticity γ on all panels
//   - viscous source distribution σ (if siglin=true)
//   - freestream
//
// Also fills state.flow.dzdg (dΨ/dγ) and state.flow.dqdm (dQ/dσ).
//
// `io`:  index of the evaluation node (0-indexed airfoil/wake), or
//         PSILIN_OFFBODY for an off-body evaluation point.

pub struct PsilinOut {
    pub psi:    f64,
    pub psi_ni: f64,
    pub qtan1:  f64,
    pub qtan2:  f64,
}

pub fn psilin(
    state: &mut XfoilState,
    io: usize,
    xi: f64, yi: f64,
    nxi: f64, nyi: f64,
    _geolin: bool,
    siglin: bool,
) -> PsilinOut {
    let n    = state.geom.n;
    let nw   = state.geom.nw;
    let seps = (state.geom.s[n - 1] - state.geom.s[0]) * 1.0e-5;

    let cosa = state.op.alfa.cos();
    let sina = state.op.alfa.sin();

    // zero sensitivity arrays
    let nz = n + nw;
    for j in 0..n  { state.flow.dzdg[j] = 0.0; state.flow.dzdn[j] = 0.0; state.flow.dqdg[j] = 0.0; }
    for j in 0..nz { state.flow.dzdm[j] = 0.0; state.flow.dqdm[j] = 0.0; }
    state.flow.z_qinf  = 0.0;
    state.flow.z_alfa  = 0.0;
    state.flow.z_qdof0 = 0.0;
    state.flow.z_qdof1 = 0.0;

    let mut psi    = 0.0f64;
    let mut psi_ni = 0.0f64;
    let mut qtan1  = 0.0f64;
    let mut qtan2  = 0.0f64;

    let (scs, sds) = if state.flow.sharp {
        (1.0, 0.0)
    } else {
        let dste = state.flow.dste.max(1e-30);
        (state.flow.ante / dste, state.flow.aste / dste)
    };

    // ── Persistent variables that must survive to the TE panel block ──────────
    // These mirror the Fortran loop variables that remain in scope after the
    // DO loop ends and execution falls through to label 11.
    let mut x1   = 0.0f64; let mut x2   = 0.0f64;
    let mut g1   = 0.0f64; let mut g2   = 0.0f64;
    let mut t1   = 0.0f64; let mut t2   = 0.0f64;
    let mut yy   = 0.0f64;
    let mut apan = 0.0f64;
    let mut x1i  = 0.0f64;            // x2i always equals x1i (Fortran lines 215-216)
    let mut yyi  = 0.0f64;
    let mut rs1  = 0.0f64; let mut rs2  = 0.0f64;
    let _ = rs1; let _ = rs2; // used inside vortex block but assigned at loop scope

    // Did the TE panel (jo=n-1) execute normally?
    // If the TE is null (sharp TE with coincident endpoints), the Fortran
    // does GO TO 12 which skips the TE contribution block entirely.
    let mut te_valid = false;

    // ── Main panel loop (Fortran DO 10 JO=1,N) ───────────────────────────────
    for jo in 0..n {
        let jp = if jo == n - 1 { 0 } else { jo + 1 };
        let jm = if jo == 0 { 0 } else { jo - 1 };
        let jq = if jo == n - 2 { jp } else { jp + 1 };

        // For the TE panel (jo == n-1): check for null/sharp TE.
        // Fortran: IF(...) GO TO 12  — skips TE contribution and freestream.
        // Here we just set te_valid=false and continue to the next iteration
        // (there is none), then check te_valid before the TE block below.
        if jo == n - 1 {
            let ds2 = (state.geom.x[jo] - state.geom.x[jp]).powi(2)
                    + (state.geom.y[jo] - state.geom.y[jp]).powi(2);
            if ds2 < seps * seps {
                // Null TE — Fortran GO TO 12 skips everything remaining
                break;
            }
        }

        let dso = ((state.geom.x[jo] - state.geom.x[jp]).powi(2)
                 + (state.geom.y[jo] - state.geom.y[jp]).powi(2)).sqrt();
        if dso == 0.0 { continue; }
        let dsio = 1.0 / dso;

        apan = state.geom.apanel[jo];

        let rx1 = xi - state.geom.x[jo];
        let ry1 = yi - state.geom.y[jo];
        let rx2 = xi - state.geom.x[jp];
        let ry2 = yi - state.geom.y[jp];

        let sx = (state.geom.x[jp] - state.geom.x[jo]) * dsio;
        let sy = (state.geom.y[jp] - state.geom.y[jo]) * dsio;

        x1 = sx * rx1 + sy * ry1;
        x2 = sx * rx2 + sy * ry2;
        yy = sx * ry1 - sy * rx1;

        rs1 = rx1 * rx1 + ry1 * ry1;
        rs2 = rx2 * rx2 + ry2 * ry2;

        // FIX BUG 1: sgn — Fortran tests IO in 1-indexed space (1..N for airfoil).
        // In 0-indexed Rust, airfoil nodes are 0..n-1. PSILIN_OFFBODY is usize::MAX.
        let sgn = if io < n { 1.0f64 } else { yy.signum() };

        (g1, t1) = if io != jo && rs1 > 0.0 {
            (rs1.ln(), (sgn * x1).atan2(sgn * yy) + (0.5 - 0.5 * sgn) * PI)
        } else {
            (0.0, 0.0)
        };

        (g2, t2) = if io != jp && rs2 > 0.0 {
            (rs2.ln(), (sgn * x2).atan2(sgn * yy) + (0.5 - 0.5 * sgn) * PI)
        } else {
            (0.0, 0.0)
        };

        // FIX BUG 2: persist x1i / yyi so TE panel block can use them.
        x1i = sx * nxi + sy * nyi;  // x2i always equals x1i
        yyi = sx * nyi - sy * nxi;

        // For the TE panel: Fortran does GO TO 11 here — skip source & vortex,
        // fall through to the TE contribution block.
        if jo == n - 1 {
            te_valid = true;
            break; // equivalent to GO TO 11
        }

        // ── Source contribution (two half-panels, Fortran SIGLIN block) ───────
        if siglin {
            let x0    = 0.5 * (x1 + x2);
            let rs0   = x0 * x0 + yy * yy;
            let g0    = rs0.ln();
            let t0    = (sgn * x0).atan2(sgn * yy) + (0.5 - 0.5 * sgn) * PI;

            // 1-0 half-panel
            let dxinv = 1.0 / (x1 - x0);
            let psum  = x0*(t0-apan) - x1*(t1-apan) + 0.5*yy*(g1-g0);
            let pdif  = ((x1+x0)*psum + rs1*(t1-apan) - rs0*(t0-apan) + (x0-x1)*yy) * dxinv;
            let psx1  = -(t1 - apan);
            let psx0  =   t0 - apan;
            let psyy  = 0.5 * (g1 - g0);
            let pdx1  = ((x1+x0)*psx1 + psum + 2.0*x1*(t1-apan) - pdif) * dxinv;
            let pdx0  = ((x1+x0)*psx0 + psum - 2.0*x0*(t0-apan) + pdif) * dxinv;
            let pdyy  = ((x1+x0)*psyy + 2.0*(x0-x1 + yy*(t1-t0))) * dxinv;

            let dsm   = ((state.geom.x[jp]-state.geom.x[jm]).powi(2)
                       + (state.geom.y[jp]-state.geom.y[jm]).powi(2)).sqrt();
            let dsim  = 1.0 / dsm;
            let ssum  = (state.panel.sig[jp]-state.panel.sig[jo])*dsio
                      + (state.panel.sig[jp]-state.panel.sig[jm])*dsim;
            let sdif  = (state.panel.sig[jp]-state.panel.sig[jo])*dsio
                      - (state.panel.sig[jp]-state.panel.sig[jm])*dsim;

            psi    += QOPI * (psum*ssum + pdif*sdif);
            let psni = psx1*x1i + psx0*(x1i+x1i)*0.5 + psyy*yyi;
            let pdni = pdx1*x1i + pdx0*(x1i+x1i)*0.5 + pdyy*yyi;
            psi_ni += QOPI * (psni*ssum + pdni*sdif);

            state.flow.dzdm[jm] += QOPI * (-psum*dsim + pdif*dsim);
            state.flow.dzdm[jo] += QOPI * (-psum*dsio - pdif*dsio);
            state.flow.dzdm[jp] += QOPI * (psum*(dsio+dsim) + pdif*(dsio-dsim));
            state.flow.dqdm[jm] += QOPI * (-psni*dsim + pdni*dsim);
            state.flow.dqdm[jo] += QOPI * (-psni*dsio - pdni*dsio);
            state.flow.dqdm[jp] += QOPI * (psni*(dsio+dsim) + pdni*(dsio-dsim));

            // 0-2 half-panel
            let dxinv2 = 1.0 / (x0 - x2);
            let psum2  = x2*(t2-apan) - x0*(t0-apan) + 0.5*yy*(g0-g2);
            let pdif2  = ((x0+x2)*psum2 + rs0*(t0-apan) - rs2*(t2-apan) + (x2-x0)*yy) * dxinv2;
            let psx02  = -(t0 - apan);
            let psx2   =   t2 - apan;
            let psyy2  = 0.5 * (g0 - g2);
            let pdx02  = ((x0+x2)*psx02 + psum2 + 2.0*x0*(t0-apan) - pdif2) * dxinv2;
            let pdx2   = ((x0+x2)*psx2  + psum2 - 2.0*x2*(t2-apan) + pdif2) * dxinv2;
            let pdyy2  = ((x0+x2)*psyy2 + 2.0*(x2-x0 + yy*(t0-t2))) * dxinv2;

            let dsp   = ((state.geom.x[jq]-state.geom.x[jo]).powi(2)
                       + (state.geom.y[jq]-state.geom.y[jo]).powi(2)).sqrt();
            let dsip  = 1.0 / dsp;
            let ssum2 = (state.panel.sig[jq]-state.panel.sig[jo])*dsip
                      + (state.panel.sig[jp]-state.panel.sig[jo])*dsio;
            let sdif2 = (state.panel.sig[jq]-state.panel.sig[jo])*dsip
                      - (state.panel.sig[jp]-state.panel.sig[jo])*dsio;

            psi    += QOPI * (psum2*ssum2 + pdif2*sdif2);
            let psni2 = psx02*(x1i+x1i)*0.5 + psx2*x1i + psyy2*yyi;
            let pdni2 = pdx02*(x1i+x1i)*0.5 + pdx2*x1i + pdyy2*yyi;
            psi_ni += QOPI * (psni2*ssum2 + pdni2*sdif2);

            state.flow.dzdm[jo] += QOPI * (-psum2*(dsip+dsio) - pdif2*(dsip-dsio));
            state.flow.dzdm[jp] += QOPI * (psum2*dsio - pdif2*dsio);
            state.flow.dzdm[jq] += QOPI * (psum2*dsip + pdif2*dsip);
            state.flow.dqdm[jo] += QOPI * (-psni2*(dsip+dsio) - pdni2*(dsip-dsio));
            state.flow.dqdm[jp] += QOPI * (psni2*dsio - pdni2*dsio);
            state.flow.dqdm[jq] += QOPI * (psni2*dsip + pdni2*dsip);
        }

        // ── Vortex panel contribution ─────────────────────────────────────────
        {
            let dxinv = 1.0 / (x1 - x2);
            let psis  = 0.5*x1*g1 - 0.5*x2*g2 + x2 - x1 + yy*(t1-t2);
            let psid  = ((x1+x2)*psis + 0.5*(rs2*g2 - rs1*g1 + x1*x1 - x2*x2)) * dxinv;

            let psx1 = 0.5 * g1;
            let psx2 = -0.5 * g2;
            let psyy = t1 - t2;
            let pdx1 = ((x1+x2)*psx1 + psis - x1*g1 - psid) * dxinv;
            let pdx2 = ((x1+x2)*psx2 + psis + x2*g2 + psid) * dxinv;
            let pdyy = ((x1+x2)*psyy - yy*(g1-g2)) * dxinv;

            let iqx   = crate::types::IQX;
            let gsum1 = state.panel.gamu[jp]      + state.panel.gamu[jo];
            let gsum2 = state.panel.gamu[jp+iqx]  + state.panel.gamu[jo+iqx];
            let gdif1 = state.panel.gamu[jp]      - state.panel.gamu[jo];
            let gdif2 = state.panel.gamu[jp+iqx]  - state.panel.gamu[jo+iqx];
            let gsum  = state.panel.gam[jp] + state.panel.gam[jo];
            let gdif  = state.panel.gam[jp] - state.panel.gam[jo];

            psi    += QOPI * (psis*gsum + psid*gdif);
            state.flow.dzdg[jo] += QOPI * (psis - psid);
            state.flow.dzdg[jp] += QOPI * (psis + psid);

            let psni = psx1*x1i + psx2*x1i + psyy*yyi; // x2i == x1i
            let pdni = pdx1*x1i + pdx2*x1i + pdyy*yyi;
            psi_ni += QOPI * (gsum*psni  + gdif*pdni);
            qtan1  += QOPI * (gsum1*psni + gdif1*pdni);
            qtan2  += QOPI * (gsum2*psni + gdif2*pdni);
            state.flow.dqdg[jo] += QOPI * (psni - pdni);
            state.flow.dqdg[jp] += QOPI * (psni + pdni);
        }
    } // end main panel loop

    // ── TE panel contribution (Fortran label 11) ──────────────────────────────
    // FIX BUG 3: only execute if the TE panel was actually processed (te_valid).
    // A null TE sets te_valid=false and we skip this block entirely (GO TO 12).
    if te_valid {
        // Variables x1, x2, yy, g1, g2, t1, t2, apan, x1i, yyi are from
        // the jo=n-1 iteration — preserved as mutable vars above.
        let jo = n - 1;
        let jp = 0usize;

        let psig = 0.5*yy*(g1-g2) + x2*(t2-apan) - x1*(t1-apan);
        let pgam = 0.5*x1*g1 - 0.5*x2*g2 + x2 - x1 + yy*(t1-t2);

        let iqx  = crate::types::IQX;
        let sigte  = 0.5*scs*(state.panel.gam[jp]     - state.panel.gam[jo]);
        let gamte  = -0.5*sds*(state.panel.gam[jp]    - state.panel.gam[jo]);
        let sigte1 = 0.5*scs*(state.panel.gamu[jp]    - state.panel.gamu[jo]);
        let sigte2 = 0.5*scs*(state.panel.gamu[jp+iqx]- state.panel.gamu[jo+iqx]);
        let gamte1 = -0.5*sds*(state.panel.gamu[jp]   - state.panel.gamu[jo]);
        let gamte2 = -0.5*sds*(state.panel.gamu[jp+iqx]-state.panel.gamu[jo+iqx]);

        psi += HOPI * (psig*sigte + pgam*gamte);
        state.flow.dzdg[jo] += -HOPI*psig*scs*0.5 + HOPI*pgam*sds*0.5;
        state.flow.dzdg[jp] +=  HOPI*psig*scs*0.5 - HOPI*pgam*sds*0.5;

        let psigx1 = -(t1 - apan);
        let psigx2 =   t2 - apan;
        let psigyy = 0.5 * (g1 - g2);
        let pgamx1 = 0.5 * g1;
        let pgamx2 = -0.5 * g2;
        let pgamyy = t1 - t2;

        // FIX BUG 2: use x1i / yyi preserved from the TE iteration.
        let psigni = psigx1*x1i + psigx2*x1i + psigyy*yyi;
        let pgamni = pgamx1*x1i + pgamx2*x1i + pgamyy*yyi;

        psi_ni += HOPI * (psigni*sigte + pgamni*gamte);
        qtan1  += HOPI * (psigni*sigte1 + pgamni*gamte1);
        qtan2  += HOPI * (psigni*sigte2 + pgamni*gamte2);
        state.flow.dqdg[jo] -= HOPI * (psigni*0.5*scs - pgamni*0.5*sds);
        state.flow.dqdg[jp] += HOPI * (psigni*0.5*scs - pgamni*0.5*sds);
    }

    // ── Freestream (Fortran label 12 falls through to here) ───────────────────
    psi    += state.op.qinf * (cosa*yi - sina*xi);
    psi_ni += state.op.qinf * (cosa*nyi - sina*nxi);
    qtan1  += state.op.qinf * nyi;
    qtan2  -= state.op.qinf * nxi;
    state.flow.z_qinf += cosa*yi - sina*xi;
    state.flow.z_alfa  -= state.op.qinf * (sina*yi + cosa*xi);

    PsilinOut { psi, psi_ni, qtan1, qtan2 }
}

// ── GGCALC ────────────────────────────────────────────────────────────────────

/// Set up the GAMU RHS vector and the Kutta/bisector special rows.
/// Called by both the full CPU ggcalc and the GPU path (after the GPU fills
/// the panel rows of AIJ, the CPU still needs to set up the RHS and special rows).
/// Fill the GAMU right-hand-side vectors and the Kutta / bisector special rows of AIJ.
/// Called internally by `ggcalc` and by the GPU path after the kernel fills the panel rows.
pub fn ggcalc_setup_rhs(state: &mut XfoilState) {
    let n   = state.geom.n;
    let nn  = n + 1;
    let iqx = crate::types::IQX;

    // zero gamma and set GAMU RHS from freestream BC
    for i in 0..n {
        state.panel.gam[i]        = 0.0;
        let yi = state.geom.y[i];
        let xi = state.geom.x[i];
        state.panel.gamu[i]       = -state.op.qinf * yi;   // α=0°
        state.panel.gamu[i + iqx] =  state.op.qinf * xi;   // α=90°
    }
    state.flow.psio = 0.0;

    // Kutta row: γ[0] + γ[N-1] = 0
    let kr = n;
    for j in 0..nn {
        state.panel.aij[kr * nn + j] = 0.0;
        state.flow.bij[kr * nn + j]  = 0.0;
    }
    state.panel.aij[kr * nn]         = 1.0; // γ at node 0
    state.panel.aij[kr * nn + n - 1] = 1.0; // γ at node N-1
    state.panel.gamu[kr]       = 0.0;
    state.panel.gamu[kr + iqx] = 0.0;

    // Sharp TE: replace last panel row with bisector velocity condition
    if state.flow.sharp {
        let xte  = 0.5 * (state.geom.x[0] + state.geom.x[n - 1]);
        let yte  = 0.5 * (state.geom.y[0] + state.geom.y[n - 1]);
        let ag1  = (-state.geom.yp[0]).atan2(-state.geom.xp[0]);
        let ag2  = (state.geom.yp[n-1]).atan2(state.geom.xp[n-1]);
        let abis = 0.5 * (ag1 + ag2);
        let cbis = abis.cos();
        let sbis = abis.sin();
        let ds1  = ((state.geom.x[0]-state.geom.x[1]).powi(2)
                  + (state.geom.y[0]-state.geom.y[1]).powi(2)).sqrt();
        let ds2  = ((state.geom.x[n-1]-state.geom.x[n-2]).powi(2)
                  + (state.geom.y[n-1]-state.geom.y[n-2]).powi(2)).sqrt();
        let xbis = xte - 0.1*ds1.min(ds2)*cbis;
        let ybis = yte - 0.1*ds1.min(ds2)*sbis;
        let _out = psilin(state, PSILIN_OFFBODY, xbis, ybis, -sbis, cbis, false, true);
        let br = n - 1;
        for j in 0..n {
            state.panel.aij[br * nn + j] = state.flow.dqdg[j];
            state.flow.bij[br * nn + j]  = -state.flow.dqdm[j];
        }
        state.panel.aij[br * nn + n] = 0.0;
        state.panel.gamu[br]       = -cbis;
        state.panel.gamu[br + iqx] = -sbis;
    }
}

/// Assemble the full panel influence matrix AIJ on the CPU and solve for the two basis vortex distributions.
/// Use `compute_panel_matrix_gpu` + `ggcalc_finish` instead for the GPU path.
pub fn ggcalc(state: &mut XfoilState) {
    let n  = state.geom.n;
    let nn = n + 1;

    // Build panel rows of AIJ via CPU psilin loop
    for i in 0..n {
        let xi  = state.geom.x[i];
        let yi  = state.geom.y[i];
        let nxi = state.geom.nx[i];
        let nyi = state.geom.ny[i];
        let _out = psilin(state, i, xi, yi, nxi, nyi, false, true);

        for j in 0..n {
            state.panel.aij[i * nn + j] = state.flow.dzdg[j];
            state.flow.bij[i * nn + j]  = -state.flow.dzdm[j];
        }
        state.panel.aij[i * nn + n] = -1.0; // dRes/dΨ₀
    }

    // Set up RHS, Kutta row, bisector row, then LU solve
    ggcalc_setup_rhs(state);
    ggcalc_finish(state);
}

/// LU-factor the already-filled AIJ matrix and back-substitute for both
/// basis solutions.  Called by ggcalc() and also by the GPU path after
/// compute_panel_matrix_gpu() fills the panel rows.
pub fn ggcalc_finish(state: &mut XfoilState) {
    let n   = state.geom.n;
    let nn  = n + 1;
    let iqx = crate::types::IQX;
    let izx = crate::types::IZX;

    ludcmp(&mut state.panel.aij, nn, nn, &mut state.panel.aij_piv);
    state.panel.aij_factored = true;

    let mut rhs1: Vec<f64> = (0..nn).map(|i| state.panel.gamu[i]).collect();
    let mut rhs2: Vec<f64> = (0..nn).map(|i| state.panel.gamu[i + iqx]).collect();
    baksub(&state.panel.aij, nn, nn, &state.panel.aij_piv, &mut rhs1);
    baksub(&state.panel.aij, nn, nn, &state.panel.aij_piv, &mut rhs2);
    for i in 0..nn {
        state.panel.gamu[i]       = rhs1[i];
        state.panel.gamu[i + iqx] = rhs2[i];
    }

    for i in 0..n {
        state.vel.qinvu[i]       = state.panel.gamu[i];
        state.vel.qinvu[i + izx] = state.panel.gamu[i + iqx];
    }
}

// ── QDCALC ───────────────────────────────────────────────────────────────────

/// Build the source-to-velocity influence matrix DIJ by back-substituting each source column through the factored AIJ.
/// Call after `ggcalc` or `ggcalc_finish`; required for viscous coupling (not used in the current inviscid solver).
pub fn qdcalc(state: &mut XfoilState) {
    assert!(state.panel.aij_factored, "qdcalc: call ggcalc first");
    let n  = state.geom.n;
    let nn = n + 1;

    if !state.panel.dij_built {
        for j in 0..n {
            let mut col: Vec<f64> = (0..nn).map(|i| state.flow.bij[i * nn + j]).collect();
            baksub(&state.panel.aij, nn, nn, &state.panel.aij_piv, &mut col);
            for i in 0..n {
                state.panel.dij[i * crate::types::IZX + j] = col[i];
            }
        }
        state.panel.dij_built = true;
    }
}

// ── CLCALC ───────────────────────────────────────────────────────────────────
// Port of xfoil.f CLCALC — integrates surface Cp to get CL and CM.
// For incompressible (Minf=0): Cp = 1 - (GAM/Qinf)²  and
//   CL = Σᵢ Cpₐᵥₑ · [(X_{i+1}-Xᵢ)cosα + (Y_{i+1}-Yᵢ)sinα]

/// Integrate the vortex-sheet strength distribution to obtain CL, CM, and CDP.
/// Returns `(CL, CM, CDP)` with Prandtl-Glauert compressibility correction applied when `minf > 0`.
pub fn clcalc(
    x: &[f64], y: &[f64], gam: &[f64],
    alfa: f64, minf: f64, qinf: f64,
    xcmref: f64, ycmref: f64,
) -> (f64, f64, f64) // (CL, CM, CDP)
{
    let n    = x.len();
    let cosa = alfa.cos();
    let sina = alfa.sin();

    // Prandtl-Glauert compressibility
    let beta = (1.0 - minf * minf).sqrt().max(1e-10);
    let bfac = 0.5 * minf * minf / (1.0 + beta);

    let cp_from_gam = |g: f64| -> f64 {
        let cginc = 1.0 - (g / qinf) * (g / qinf);
        cginc / (beta + bfac * cginc)
    };

    let (mut cl, mut cm, mut cdp) = (0.0f64, 0.0f64, 0.0f64);
    let mut cpg1 = cp_from_gam(gam[0]);

    for i in 0..n {
        let ip   = if i == n - 1 { 0 } else { i + 1 };
        let cpg2 = cp_from_gam(gam[ip]);

        let dx = (x[ip] - x[i]) * cosa + (y[ip] - y[i]) * sina;
        let dy = (y[ip] - y[i]) * cosa - (x[ip] - x[i]) * sina;
        let dg = cpg2 - cpg1;

        let ax = (0.5*(x[ip]+x[i]) - xcmref)*cosa + (0.5*(y[ip]+y[i]) - ycmref)*sina;
        let ay = (0.5*(y[ip]+y[i]) - ycmref)*cosa - (0.5*(x[ip]+x[i]) - xcmref)*sina;
        let ag = 0.5 * (cpg2 + cpg1);

        cl  += dx * ag;
        cdp -= dy * ag;
        cm  -= dx * (ag*ax + dg*dx/12.0) + dy * (ag*ay + dg*dy/12.0);

        cpg1 = cpg2;
    }
    (cl, cm, cdp)
}

// ── SPECAL ───────────────────────────────────────────────────────────────────

/// Superpose the two basis vortex solutions for the current angle of attack and compute CL, CM, CDP, and the Cp distribution.
/// Call after `ggcalc_finish` each time `state.op.alfa` changes.
pub fn specal(state: &mut XfoilState) {
    let n    = state.geom.n;
    let iqx  = crate::types::IQX;
    let izx  = crate::types::IZX;
    let cosa = state.op.alfa.cos();
    let sina = state.op.alfa.sin();

    // Superpose α=0° and α=90° basis solutions
    for i in 0..n {
        state.panel.gam[i] = cosa * state.panel.gamu[i]
                           + sina * state.panel.gamu[i + iqx];
    }

    // Store inviscid surface speeds in QINVU (basis) and QINV (current α)
    for i in 0..n {
        state.vel.qinvu[i]       = state.panel.gamu[i];
        state.vel.qinvu[i + izx] = state.panel.gamu[i + iqx];
        state.vel.qinv[i] = cosa * state.vel.qinvu[i]
                          + sina * state.vel.qinvu[i + izx];
    }

    // CL and CM via pressure integration (CLCALC)
    let qinf  = state.op.qinf.max(1e-30);
    let _chord = state.geom.chord.max(1e-30); // retained for future CM normalisation
    let gam_slice: Vec<f64> = state.panel.gam[..n].to_vec();
    let x_slice:   Vec<f64> = state.geom.x[..n].to_vec();
    let y_slice:   Vec<f64> = state.geom.y[..n].to_vec();
    let (cl, cm, cdp) = clcalc(
        &x_slice, &y_slice, &gam_slice,
        state.op.alfa, state.op.minf, qinf,
        0.25, 0.0, // standard quarter-chord reference
    );
    state.op.cl  = cl;
    state.op.cm  = cm;
    state.op.cdp = cdp;

    // Circulation (Kutta-Joukowski cross-check, not used for CL)
    let mut circ = 0.0f64;
    for i in 0..n - 1 {
        let ds = ((x_slice[i+1]-x_slice[i]).powi(2)
                + (y_slice[i+1]-y_slice[i]).powi(2)).sqrt();
        circ += 0.5 * (gam_slice[i] + gam_slice[i+1]) * ds;
    }
    state.op.circ = circ;

    // Inviscid Cp
    for i in 0..n {
        state.vel.cpi[i] = 1.0 - (state.vel.qinv[i] / qinf).powi(2);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::XfoilState;
    use crate::geometry::load_naca;
    use std::f64::consts::PI;

    #[test]
    fn calc_normals_unit_length() {
        let mut state = XfoilState::default();
        load_naca(&mut state, "0012");
        calc_normals(&mut state);
        for i in 0..state.geom.n {
            let mag = (state.geom.nx[i].powi(2) + state.geom.ny[i].powi(2)).sqrt();
            assert!((mag - 1.0).abs() < 1e-10, "normal[{i}] mag={mag}");
        }
    }

    #[test]
    fn circle_surface_speed() {
        // A circle has the exact analytical solution: Ue = 2*Qinf*sin(θ).
        // At the top (θ=π/2, x=0.5, y=0.5): Ue = 2.0 (with Qinf=1).
        // If gamu[:,0] gives Ue at the top ≈ 2.0 → panel solve is correct.
        // If ≈ 1.0 → AIJ is 2× too large (factor-of-2 bug confirmed).
        use crate::types::XfoilState;
        use crate::spline::scalc;

        let n = 64usize; // 64 panels around circle
        let r = 0.5f64;  // radius 0.5, centered at (0.5, 0)

        // Build circle coordinates going counterclockwise: TE at (1,0), up over top, back to TE
        let mut xb = vec![0.0f64; n];
        let mut yb = vec![0.0f64; n];
        for i in 0..n {
            let t = 2.0 * PI * (i as f64) / (n as f64);
            xb[i] = 0.5 + r * t.cos();
            yb[i] = r * t.sin();
        }
        // close the loop (last panel goes back to first)

        let mut state = XfoilState::default();
        state.geom.n = n;
        state.geom.x[..n].copy_from_slice(&xb);
        state.geom.y[..n].copy_from_slice(&yb);
        state.op.qinf = 1.0;
        state.op.alfa = 0.0;
        state.flow.sharp = true; // circle has coincident TE points

        // compute arc length + spline
        scalc(&xb, &yb, &mut state.geom.s[..n]);
        state.geom.chord = 1.0;
        state.geom.sle = state.geom.s[n/2]; // approximate
        state.geom.xle = xb[n/2];
        state.geom.yle = yb[n/2];
        state.geom.xte = xb[0];
        state.geom.yte = yb[0];

        calc_normals(&mut state);
        calc_panel_angles(&mut state);
        ggcalc(&mut state);

        // Node at the top of the circle: i ≈ n/4 (θ=π/2)
        let i_top = n / 4;
        let ue_top = state.panel.gamu[i_top].abs(); // surface speed at top
        eprintln!("Circle top node {i_top}: x={:.3} y={:.3} gamu0={:.4}",
            xb[i_top], yb[i_top], state.panel.gamu[i_top]);
        // Exact: Ue = 2*sin(π/2) = 2.0 for α=0° (x-flow past circle)
        // Tolerance 15% for N=64 discretization
        assert!(ue_top > 1.5 && ue_top < 2.5,
            "Circle top speed={ue_top:.4}, expected ~2.0");
    }

    #[test]
    fn debug_gam_values() {
        let mut state = XfoilState::default();
        load_naca(&mut state, "0012");
        state.op.alfa = 0.0;
        state.op.qinf = 1.0;
        calc_normals(&mut state);
        calc_panel_angles(&mut state);
        ggcalc(&mut state);
        let n = state.geom.n;
        // For symmetric airfoil at α=0, gamu[i] (basis α=0) should be ≈ 1.0 (free-stream speed)
        // Sample 5 values near the leading edge area
        for i in [n/4, n/3, n/2, 2*n/3, 3*n/4] {
            let gamu0 = state.panel.gamu[i];
            let gamu1 = state.panel.gamu[i + crate::types::IQX];
            eprintln!("gamu[{i}] = ({gamu0:.4}, {gamu1:.4})  x={:.3}", state.geom.x[i]);
        }
        // Check the Kutta condition row result (should be Ψ₀)
        eprintln!("gamu[n]={:.4} (Psi0)", state.panel.gamu[n]);
    }

    #[test]
    fn inviscid_naca0012_cl_at_5deg() {
        // NACA 0012 at α=5° should give CL ≈ 0.55 (thin-airfoil theory: 2π·sin5° ≈ 0.548)
        let mut state = XfoilState::default();
        load_naca(&mut state, "0012");
        state.op.alfa  = 5.0 * PI / 180.0;
        state.op.qinf  = 1.0;
        state.op.reinf = 1e6;
        state.op.minf  = 0.0;

        calc_normals(&mut state);
        calc_panel_angles(&mut state);
        ggcalc(&mut state);
        specal(&mut state);

        let cl = state.op.cl;
        assert!(cl.is_finite(), "CL is not finite: {cl}");
        assert!(cl > 0.4 && cl < 0.75,
            "CL={cl:.4} out of expected range [0.40, 0.75] for NACA 0012 at 5°");
    }

    #[test]
    fn inviscid_naca0012_zero_alpha_zero_cl() {
        // Symmetric airfoil at α=0 → CL = 0
        let mut state = XfoilState::default();
        load_naca(&mut state, "0012");
        state.op.alfa = 0.0;
        state.op.qinf = 1.0;

        calc_normals(&mut state);
        calc_panel_angles(&mut state);
        ggcalc(&mut state);
        specal(&mut state);

        let cl = state.op.cl;
        assert!(cl.is_finite(), "CL is not finite: {cl}");
        assert!(cl.abs() < 0.05,
            "CL={cl:.5} should be ~0 for symmetric airfoil at α=0");
    }

    #[test]
    fn inviscid_cl_scales_with_alpha() {
        // CL should scale roughly linearly with α in the attached regime
        let mut s1 = XfoilState::default();
        let mut s2 = XfoilState::default();
        load_naca(&mut s1, "0012");
        load_naca(&mut s2, "0012");
        s1.op.alfa = 3.0 * PI / 180.0; s1.op.qinf = 1.0;
        s2.op.alfa = 6.0 * PI / 180.0; s2.op.qinf = 1.0;

        for s in [&mut s1, &mut s2] {
            calc_normals(s); calc_panel_angles(s); ggcalc(s); specal(s);
        }

        let cl1 = s1.op.cl;
        let cl2 = s2.op.cl;
        assert!(cl1.is_finite() && cl2.is_finite());
        // CL at 6° should be roughly 2× CL at 3°
        let ratio = cl2 / cl1;
        assert!(ratio > 1.7 && ratio < 2.3,
            "CL ratio (6°/3°) = {ratio:.3}, expected ~2.0");
    }

    #[test]
    fn inviscid_naca4412_positive_cl_at_zero() {
        // Cambered airfoil has positive CL at α=0
        let mut state = XfoilState::default();
        load_naca(&mut state, "4412");
        state.op.alfa = 0.0;
        state.op.qinf = 1.0;

        calc_normals(&mut state);
        calc_panel_angles(&mut state);
        ggcalc(&mut state);
        specal(&mut state);

        let cl = state.op.cl;
        assert!(cl.is_finite());
        assert!(cl > 0.2, "NACA 4412 at α=0 should have CL > 0.2, got {cl:.4}");
    }
}
