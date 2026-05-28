//! C FFI layer for TitaniumFoil.
//!
//! Exposes a plain C API so the library can be called from C++ (or any other
//! language with a C FFI).  The C++ header `include/titaniumfoil.hpp` wraps
//! these functions in an ergonomic RAII class.

use std::ffi::CStr;
use std::os::raw::c_char;
use tf::{Solver, Point};

// ── C-compatible result struct ────────────────────────────────────────────────

/// Aerodynamic coefficients at one operating point.
/// Matches the layout of `titaniumfoil::Point` exactly.
#[repr(C)]
pub struct TfPoint {
    pub alpha: f64,
    pub cl:    f64,
    pub cd:    f64,
    pub cdp:   f64,
    pub cdf:   f64,
    pub cm:    f64,
    pub ld:    f64,
}

impl From<Point> for TfPoint {
    fn from(p: Point) -> Self {
        Self { alpha: p.alpha, cl: p.cl, cd: p.cd, cdp: p.cdp,
               cdf: p.cdf, cm: p.cm, ld: p.ld }
    }
}

// ── Polar result (heap-allocated Vec) ────────────────────────────────────────

pub struct TfPolar(Vec<Option<Point>>);

// ── Solver lifecycle ──────────────────────────────────────────────────────────

/// Create a new Solver.  `nside` = panel nodes per surface (65 = default).
/// Caller owns the pointer — free with `tf_solver_free`.
#[no_mangle]
pub extern "C" fn tf_solver_new(nside: usize) -> *mut Solver {
    Box::into_raw(Box::new(Solver::with_panels(nside)))
}

/// Free a Solver created by `tf_solver_new`.
#[no_mangle]
pub extern "C" fn tf_solver_free(solver: *mut Solver) {
    if !solver.is_null() {
        unsafe { drop(Box::from_raw(solver)); }
    }
}

// ── Single operating point ────────────────────────────────────────────────────

/// Compute one operating point.
///
/// Returns 1 on success and writes into `*out`.
/// Returns 0 if the NACA designation is invalid or the solve diverges.
#[no_mangle]
pub extern "C" fn tf_analyze(
    solver:    *const Solver,
    naca:      *const c_char,
    alpha_deg: f64,
    re:        f64,
    out:       *mut TfPoint,
) -> i32 {
    if solver.is_null() || naca.is_null() || out.is_null() { return 0; }
    let naca_str = unsafe { CStr::from_ptr(naca) }.to_string_lossy();
    let solver   = unsafe { &*solver };
    match solver.analyze(&naca_str, alpha_deg, re) {
        Ok(Some(p)) => { unsafe { *out = p.into(); } 1 }
        _           => 0,
    }
}

// ── Polar sweep ───────────────────────────────────────────────────────────────

/// Compute a polar sweep.  Panel matrix built once; `specal` runs per alpha.
///
/// `alphas`   pointer to array of `n` angles of attack in degrees.
/// Returns a `TfPolar*` — free with `tf_polar_free`.
#[no_mangle]
pub extern "C" fn tf_polar(
    solver: *const Solver,
    naca:   *const c_char,
    alphas: *const f64,
    n:      usize,
    re:     f64,
) -> *mut TfPolar {
    if solver.is_null() || naca.is_null() || alphas.is_null() {
        return std::ptr::null_mut();
    }
    let naca_str = unsafe { CStr::from_ptr(naca) }.to_string_lossy();
    let alphas   = unsafe { std::slice::from_raw_parts(alphas, n) };
    let solver   = unsafe { &*solver };
    match solver.polar(&naca_str, alphas, re) {
        Ok(pts) => Box::into_raw(Box::new(TfPolar(pts))),
        Err(_)  => std::ptr::null_mut(),
    }
}

/// Number of entries in a polar result.
#[no_mangle]
pub extern "C" fn tf_polar_len(polar: *const TfPolar) -> usize {
    if polar.is_null() { return 0; }
    unsafe { (*polar).0.len() }
}

/// Get one entry from a polar result.
/// Returns 1 and writes into `*out` if the point converged, 0 if it diverged.
#[no_mangle]
pub extern "C" fn tf_polar_get(
    polar: *const TfPolar,
    index: usize,
    out:   *mut TfPoint,
) -> i32 {
    if polar.is_null() || out.is_null() { return 0; }
    let polar = unsafe { &*polar };
    match polar.0.get(index) {
        Some(Some(p)) => { unsafe { *out = p.clone().into(); } 1 }
        _             => 0,
    }
}

/// Free a polar result created by `tf_polar`.
#[no_mangle]
pub extern "C" fn tf_polar_free(polar: *mut TfPolar) {
    if !polar.is_null() {
        unsafe { drop(Box::from_raw(polar)); }
    }
}

// ── Multi-Re polar ────────────────────────────────────────────────────────────

/// Compute a polar for multiple Reynolds numbers simultaneously.
///
/// Returns a flat row-major array: `result[re_idx * n_alpha + alpha_idx]`.
/// `valid` is filled with 1 where the point converged, 0 where it diverged.
/// Caller must free both `result` and `valid` with `tf_free_f64` / `tf_free_i32`.
#[no_mangle]
pub extern "C" fn tf_polar_multi_re(
    solver:   *const Solver,
    naca:     *const c_char,
    alphas:   *const f64,
    n_alpha:  usize,
    res:      *const f64,
    n_re:     usize,
    out_pts:  *mut *mut TfPoint,
    out_valid: *mut *mut i32,
) -> i32 {
    if solver.is_null() || naca.is_null() || alphas.is_null() || res.is_null() { return 0; }
    let naca_str  = unsafe { CStr::from_ptr(naca) }.to_string_lossy();
    let alphas    = unsafe { std::slice::from_raw_parts(alphas, n_alpha) };
    let res_slice = unsafe { std::slice::from_raw_parts(res, n_re) };
    let solver    = unsafe { &*solver };

    let grid = match solver.polar_multi_re(&naca_str, alphas, res_slice) {
        Ok(g)  => g,
        Err(_) => return 0,
    };

    let total = n_re * n_alpha;
    let mut pts:   Vec<TfPoint> = Vec::with_capacity(total);
    let mut valid: Vec<i32>     = Vec::with_capacity(total);

    for row in &grid {
        for opt in row {
            match opt {
                Some(p) => { pts.push(p.clone().into()); valid.push(1); }
                None    => { pts.push(TfPoint { alpha:0.0,cl:0.0,cd:0.0,cdp:0.0,cdf:0.0,cm:0.0,ld:0.0 }); valid.push(0); }
            }
        }
    }

    pts.shrink_to_fit();
    valid.shrink_to_fit();
    unsafe {
        *out_pts   = pts.as_mut_ptr();
        *out_valid = valid.as_mut_ptr();
        std::mem::forget(pts);
        std::mem::forget(valid);
    }
    1
}

/// Free a TfPoint array returned by `tf_polar_multi_re`.
#[no_mangle]
pub extern "C" fn tf_free_points(ptr: *mut TfPoint, len: usize) {
    if !ptr.is_null() {
        unsafe { drop(Vec::from_raw_parts(ptr, len, len)); }
    }
}

/// Free an i32 array returned by `tf_polar_multi_re`.
#[no_mangle]
pub extern "C" fn tf_free_i32(ptr: *mut i32, len: usize) {
    if !ptr.is_null() {
        unsafe { drop(Vec::from_raw_parts(ptr, len, len)); }
    }
}
