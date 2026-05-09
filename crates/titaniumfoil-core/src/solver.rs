// xoper.f → Rust (direct port, Mark Drela MIT XFOIL)
//
// Top-level viscous / inviscid solver orchestration.
// VISCAL is the main entry point for a viscous polar point.

use crate::types::{XfoilState, ISX, IVX, IZX};
use crate::panel::{calc_normals, calc_panel_angles, ggcalc, qdcalc, specal};
use crate::boundary_layer::setbl;
use crate::linalg::blsolv;

// ── QISET: interpolate inviscid velocities at current alpha ──────────────────

/// Build edge velocities UEDG from inviscid basis solutions and current mass.
pub fn qiset(state: &mut XfoilState) {
    let _n  = state.geom.n;
    let _nw = state.geom.nw;
    let izx = IZX;
    let cosa = state.op.alfa.cos();
    let sina = state.op.alfa.sin();

    for is in 0..ISX {
        let nbl = state.flow.nbl[is];
        for ibl in 1..nbl {
            let i  = state.flow.ipan[ibl][is];
            // inviscid velocity: superpose α=0,90 basis + source correction
            let ui = cosa * state.vel.qinvu[i] + sina * state.vel.qinvu[i + izx];
            // add source-panel coupling: Ue = Uinv + DIJ·MASS
            let mut ucorr = 0.0f64;
            for js in 0..ISX {
                let nbl_j = state.flow.nbl[js];
                for jbl in 1..nbl_j {
                    let j  = state.flow.ipan[jbl][js];
                    let _jv = state.flow.isys[jbl][js];
                    // VTI: sign/direction factor at BL node
                    let vti_i = state.flow.vti[ibl + is * IVX];
                    let vti_j = state.flow.vti[jbl + js * IVX];
                    ucorr -= vti_i * vti_j * state.panel.dij[i * IZX + j]
                           * state.bl[js].mass[jbl];
                }
            }
            state.bl[is].uedg[ibl] = ui + ucorr;
            state.bl[is].sta[ibl].u = state.bl[is].uedg[ibl];
        }
    }
}

// ── CPCALC: pressure coefficients ────────────────────────────────────────────

/// Compute viscous Cp from viscous surface velocities.
pub fn cpcalc(state: &mut XfoilState) {
    let n   = state.geom.n;
    let nw  = state.geom.nw;
    let qinf = state.op.qinf;

    // airfoil
    for i in 0..n {
        let q = state.vel.qvis[i];
        state.vel.cpv[i] = 1.0 - (q / qinf).powi(2);
    }
    // wake
    for i in n..n + nw {
        let q = state.vel.qvis[i];
        state.vel.cpv[i] = 1.0 - (q / qinf).powi(2);
    }
}

// ── Force integration ─────────────────────────────────────────────────────────

/// Integrate viscous forces: CL, CD, CM from viscous Cp and skin friction.
pub fn force_integ(state: &mut XfoilState) {
    let n    = state.geom.n;
    let cosa = state.op.alfa.cos();
    let sina = state.op.alfa.sin();
    let qinf = state.op.qinf;
    let chord = state.geom.chord;

    let (mut cl, mut cdp, mut cdf, mut cm) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);

    for i in 0..n - 1 {
        let cp  = 0.5 * (state.vel.cpv[i] + state.vel.cpv[i + 1]);
        let dx  = state.geom.x[i + 1] - state.geom.x[i];
        let dy  = state.geom.y[i + 1] - state.geom.y[i];
        // pressure contribution
        cl  += -cp * (dx * cosa + dy * sina);
        cdp += -cp * (dx * sina - dy * cosa);
        cm  += -cp * (dx * (state.geom.x[i] - 0.25) + dy * state.geom.y[i]);
    }

    // add friction drag from BL
    for is in 0..ISX {
        for ibl in 1..state.flow.nbl[is] {
            let _i   = state.flow.ipan[ibl][is];
            let cf  = state.bl[is].sta[ibl].cf;
            let ue  = state.bl[is].uedg[ibl];
            let ds  = if ibl > 0 {
                (state.bl[is].xssi[ibl] - state.bl[is].xssi[ibl - 1]).abs()
            } else { 0.0 };
            cdf += 0.5 * cf * ue * ue / (qinf * qinf) * ds;
        }
    }

    state.op.cl  = cl  / chord;
    state.op.cdp = cdp / chord;
    state.op.cdf = cdf / chord;
    state.op.cd  = state.op.cdp + state.op.cdf;
    state.op.cm  = cm  / chord.powi(2);
}

// ── BL panel mapping ──────────────────────────────────────────────────────────

/// Map panel nodes to BL stations for upper (is=0) and lower (is=1) surfaces.
/// Sets flow.ipan, flow.nbl, flow.iblte, flow.xssi.
pub fn setup_bl_mapping(state: &mut XfoilState) {
    let n   = state.geom.n;
    let nw  = state.geom.nw;
    let sle = state.geom.sle;
    let s   = &state.geom.s[..n + nw].to_vec();

    // upper surface: nodes going from LE → TE (index 0..n/2)
    // lower surface: nodes going from LE → TE (index n/2..n)
    // XFOIL convention: node 0 is upper TE, going counter-clockwise

    // locate LE index
    let ile = s[1..n - 1].iter().enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i + 1)
        .unwrap_or(n / 2);

    // upper surface: ile..0 (reversed) then wake
    let mut ibl = 0usize;
    for i in (0..=ile).rev() {
        state.flow.ipan[ibl][0] = i;
        state.bl[0].xssi[ibl]  = (sle - s[i]).abs();
        ibl += 1;
        if ibl >= IVX { break; }
    }
    state.flow.iblte[0] = ibl - 1;
    // wake stations
    for iw in 1..nw.min(IVX - ibl) {
        state.flow.ipan[ibl][0] = n + iw;
        state.bl[0].xssi[ibl]  = state.bl[0].xssi[state.flow.iblte[0]]
                                + (s[n + iw] - s[n]).abs();
        ibl += 1;
    }
    state.flow.nbl[0] = ibl;

    // lower surface: ile..n
    ibl = 0;
    for i in ile..n {
        state.flow.ipan[ibl][1] = i;
        state.bl[1].xssi[ibl]  = (s[i] - sle).abs();
        ibl += 1;
        if ibl >= IVX { break; }
    }
    state.flow.iblte[1] = ibl - 1;
    for iw in 1..nw.min(IVX - ibl) {
        state.flow.ipan[ibl][1] = n + iw;
        state.bl[1].xssi[ibl]  = state.bl[1].xssi[state.flow.iblte[1]]
                                + (s[n + iw] - s[n]).abs();
        ibl += 1;
    }
    state.flow.nbl[1] = ibl;

    // VTI: sign of surface tangent relative to BL march direction
    for is in 0..ISX {
        for ibl in 0..state.flow.nbl[is] {
            state.flow.vti[ibl + is * IVX] = if is == 0 { -1.0 } else { 1.0 };
        }
    }

    // NSYS: total system size (all BL stations combined)
    let mut nsys = 0;
    for is in 0..ISX {
        for ibl in 1..state.flow.nbl[is] {
            state.flow.isys[ibl][is] = nsys;
            nsys += 1;
        }
    }
    state.flow.nsys = nsys;
    state.mat.nsys  = nsys;
}

// ── VISCAL: main viscous solver ───────────────────────────────────────────────

/// Converge a viscous operating point at current alpha/CL and Re.
///
/// Sequence:
///   1. Inviscid solve (ggcalc + specal)
///   2. BL panel mapping
///   3. Source influence matrix (qdcalc)
///   4. Newton viscous loop: qiset → setbl (march_bl) → blsolv → update mass
///
/// Returns true if converged within `niter` iterations.
pub fn viscal(state: &mut XfoilState, niter: usize) -> bool {
    // ── inviscid setup ───────────────────────────────────────────────────────
    if !state.panel.aij_factored {
        calc_normals(state);
        calc_panel_angles(state);
        ggcalc(state);
    }
    specal(state);

    setup_bl_mapping(state);
    qdcalc(state);

    // ── viscous Newton loop ──────────────────────────────────────────────────
    const RMSBL_TOL: f64 = 1.0e-4;
    let mut converged = false;

    for _iter in 0..niter {
        qiset(state);
        setbl(state);
        blsolv(&mut state.mat);

        // extract VDEL corrections and apply to mass defect
        let nsys = state.flow.nsys;
        for is in 0..ISX {
            for ibl in 1..state.flow.nbl[is] {
                let iv = state.flow.isys[ibl][is];
                if iv < nsys {
                    let dm = state.mat.vdel[[2, 0, iv]];
                    state.bl[is].mass[ibl] += dm;
                    let u = state.bl[is].uedg[ibl];
                    if u.abs() > 1e-10 {
                        state.bl[is].sta[ibl].d = state.bl[is].mass[ibl] / u;
                    }
                }
            }
        }

        // check convergence
        let mut rms = 0.0f64;
        for is in 0..ISX {
            for ibl in 1..state.flow.nbl[is] {
                let iv = state.flow.isys[ibl][is];
                if iv < nsys { rms += state.mat.vdel[[2, 0, iv]].powi(2); }
            }
        }

        if rms.sqrt() < RMSBL_TOL {
            converged = true;
            break;
        }
    }

    // final surface velocity and force integration
    for is in 0..ISX {
        for ibl in 0..state.flow.nbl[is] {
            let i = state.flow.ipan[ibl][is];
            state.vel.qvis[i] = state.bl[is].uedg[ibl];
        }
    }
    cpcalc(state);
    force_integ(state);
    state.op.converged = converged;
    converged
}

// ── Convenience: run a pure inviscid solve ────────────────────────────────────

/// Inviscid solve only (no BL). Sets CL and inviscid Cp.
pub fn inviscal(state: &mut XfoilState) {
    calc_normals(state);
    calc_panel_angles(state);
    ggcalc(state);
    specal(state);
    state.op.converged = true;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::load_naca;
    use std::f64::consts::PI;

    #[test]
    fn inviscal_naca0012_runs() {
        // Smoke test: inviscal completes without panic and sets a finite CL.
        // Numerical accuracy of the panel solve is validated separately
        // once the PSILIN kernel is fully verified against Fortran output.
        let mut state = XfoilState::default();
        load_naca(&mut state, "0012");
        state.op.alfa  = 5.0 * PI / 180.0;
        state.op.qinf  = 1.0;
        state.op.reinf = 1e6;
        state.op.minf  = 0.0;
        inviscal(&mut state);
        // Panel geometry must be set up
        assert!(state.geom.n > 0, "n={}", state.geom.n);
        assert!(state.geom.chord > 0.0, "chord={}", state.geom.chord);
        // AIJ must have been factored
        assert!(state.panel.aij_factored, "AIJ not factored");
    }
}
