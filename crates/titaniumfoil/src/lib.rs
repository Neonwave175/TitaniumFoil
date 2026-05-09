//! TitaniumFoil — GPU-accelerated NACA airfoil analysis
//!
//! # Quick start
//!
//! ```no_run
//! use titaniumfoil::Solver;
//!
//! let solver = Solver::new();
//!
//! // Single operating point
//! if let Ok(Some(r)) = solver.analyze("4412", 4.0, 200_000.0) {
//!     println!("CL={:.4}  CD={:.5}  L/D={:.1}", r.cl, r.cd, r.ld);
//! }
//!
//! // Full polar (panel matrix built once — fast)
//! let polar = solver.polar("63412", &[-4.0,-2.0,0.0,2.0,4.0,6.0], 150_000.0).unwrap();
//! for p in polar.iter().flatten() { println!("α={:.0}°  L/D={:.1}", p.alpha, p.ld); }
//! ```
//!
//! # Performance
//!
//! On Apple Silicon the GPU panel solve takes ~0.5 ms at N=129 (nside=65).
//! For highest throughput use [`Solver::polar`] (panel matrix shared across α)
//! or parallelize across airfoils with rayon — each thread gets its own GPU
//! context automatically.
//!
//! ```no_run
//! use titaniumfoil::Solver;
//! use rayon::prelude::*;
//!
//! let solver = Solver::new();
//! let airfoils = vec!["4412", "2412", "63412", "65415"];
//!
//! let results: Vec<_> = airfoils.par_iter()
//!     .map(|naca| solver.polar(naca, &[0.0, 2.0, 4.0, 6.0], 200_000.0))
//!     .collect();
//! ```

use std::cell::RefCell;
use std::f64::consts::PI;

use titaniumfoil_core::types::XfoilState;
use titaniumfoil_core::geometry::load_naca_n;
use titaniumfoil_core::panel::{calc_normals, calc_panel_angles, ggcalc_finish, specal};
use titaniumfoil_core::viscous::skin_friction_drag;
use titaniumfoil_metal::context::MetalContext;
use titaniumfoil_metal::panel_gpu::{compute_panel_matrix_gpu, compute_panel_matrix_batch_gpu};

// One GPU context per thread — each rayon worker gets its own command queue,
// enabling fully concurrent GPU dispatches with no mutex contention.
thread_local! {
    static GPU: RefCell<MetalContext> = RefCell::new(MetalContext::new());
}

// ── Result type ───────────────────────────────────────────────────────────────

/// Aerodynamic coefficients at a single operating point.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    /// Angle of attack (degrees)
    pub alpha: f64,
    /// Lift coefficient
    pub cl:    f64,
    /// Total drag coefficient  (pressure + skin friction)
    pub cd:    f64,
    /// Pressure drag coefficient
    pub cdp:   f64,
    /// Skin-friction drag coefficient (Thwaites + Head integral BL)
    pub cdf:   f64,
    /// Pitching moment coefficient about quarter-chord
    pub cm:    f64,
    /// Lift-to-drag ratio  (CL / CD)
    pub ld:    f64,
}

impl Point {
    fn from_state(state: &XfoilState, alpha_deg: f64) -> Option<Self> {
        let cl  = state.op.cl;
        let cdp = state.op.cdp;
        let cdf = skin_friction_drag(state);
        let cd  = cdp + cdf;
        let cm  = state.op.cm;
        if !cl.is_finite() || !cd.is_finite() || cd < 1e-9 { return None; }
        Some(Self { alpha: alpha_deg, cl, cd, cdp, cdf, cm, ld: cl / cd })
    }
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors returned by [`Solver`].
#[derive(Debug)]
pub enum Error {
    /// Unknown or unsupported NACA designation.
    UnknownAirfoil(String),
    /// Panel count would exceed IQX=360.
    PanelCountExceeded,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownAirfoil(s) => write!(f, "unknown NACA designation: {s}"),
            Self::PanelCountExceeded => write!(f, "panel count exceeds IQX=360"),
        }
    }
}
impl std::error::Error for Error {}

// ── Solver ────────────────────────────────────────────────────────────────────

/// Reusable airfoil solver.
///
/// `Solver` is cheap to clone and is `Send + Sync` — share one instance
/// across threads or create one per thread.  The underlying GPU context is
/// managed per-thread via `thread_local!`, so no locking is needed.
///
/// # Panel resolution (`nside`)
///
/// | nside | N panels | speed       | accuracy       |
/// |-------|----------|-------------|----------------|
/// | 65    | 129      | ~21 000/s   | fast survey    |
/// | 120   | 239      | ~7 500/s    | good           |
/// | 180   | 359      | ~3 000/s    | near-XFOIL     |
#[derive(Debug, Clone)]
pub struct Solver {
    /// Number of panel nodes per surface (total N = 2·nside − 1).
    pub nside: usize,
}

impl Default for Solver {
    fn default() -> Self { Self::new() }
}

impl Solver {
    /// Create a solver with the default panel resolution (nside=65, N=129).
    pub fn new() -> Self { Self { nside: 65 } }

    /// Create a solver with a specific number of panel nodes per surface.
    pub fn with_panels(nside: usize) -> Self { Self { nside } }

    // ── Single operating point ────────────────────────────────────────────────

    /// Compute aerodynamic coefficients for one airfoil at one α and Re.
    ///
    /// Returns `None` if the solver diverges (e.g. extreme geometry).
    ///
    /// ```no_run
    /// # use titaniumfoil::Solver;
    /// let p = Solver::new().analyze("4412", 4.0, 200_000.0).unwrap().unwrap();
    /// assert!(p.cl > 0.5);
    /// ```
    pub fn analyze(&self, naca: &str, alpha_deg: f64, re: f64) -> Result<Option<Point>, Error> {
        let mut state = self.build_panel_state(naca, re)?;
        GPU.with(|ctx| compute_panel_matrix_gpu(&ctx.borrow(), &mut state));
        ggcalc_finish(&mut state);
        state.op.alfa = alpha_deg * PI / 180.0;
        specal(&mut state);
        Ok(Point::from_state(&state, alpha_deg))
    }

    // ── Polar sweep ───────────────────────────────────────────────────────────

    /// Compute a polar for one airfoil across multiple angles of attack.
    ///
    /// The O(N²) GPU panel solve runs **once**; `specal` (O(N)) runs per α.
    /// Use this instead of calling [`analyze`](Self::analyze) in a loop.
    ///
    /// Returns one `Option<Point>` per α — `None` for diverged angles.
    ///
    /// ```no_run
    /// # use titaniumfoil::Solver;
    /// let solver = Solver::new();
    /// let alphas: Vec<f64> = (-5..=15).map(|a| a as f64).collect();
    /// let polar = solver.polar("4412", &alphas, 200_000.0).unwrap();
    /// let best = polar.into_iter().flatten()
    ///     .max_by(|a, b| a.ld.partial_cmp(&b.ld).unwrap()).unwrap();
    /// println!("best L/D = {:.1}", best.ld);
    /// ```
    pub fn polar(&self, naca: &str, alphas: &[f64], re: f64)
        -> Result<Vec<Option<Point>>, Error>
    {
        let mut state = self.build_panel_state(naca, re)?;
        GPU.with(|ctx| compute_panel_matrix_gpu(&ctx.borrow(), &mut state));
        ggcalc_finish(&mut state);
        Ok(alphas.iter().map(|&alpha| {
            state.op.alfa = alpha * PI / 180.0;
            specal(&mut state);
            Point::from_state(&state, alpha)
        }).collect())
    }

    // ── Multi-Re polar ────────────────────────────────────────────────────────

    /// Compute a polar across multiple Re values, sharing the panel matrix.
    ///
    /// `specal` is Re-independent, so it runs once per α; `skin_friction_drag`
    /// runs once per (α, Re) pair.
    ///
    /// Returns `results[re_idx][alpha_idx]`.
    pub fn polar_multi_re(
        &self,
        naca:   &str,
        alphas: &[f64],
        res:    &[f64],
    ) -> Result<Vec<Vec<Option<Point>>>, Error> {
        let mut state = self.build_panel_state(naca, res[0])?;
        GPU.with(|ctx| compute_panel_matrix_gpu(&ctx.borrow(), &mut state));
        ggcalc_finish(&mut state);

        let mut out: Vec<Vec<Option<Point>>> = vec![Vec::new(); res.len()];
        for &alpha in alphas {
            state.op.alfa = alpha * PI / 180.0;
            specal(&mut state);
            let cl  = state.op.cl;
            let cdp = state.op.cdp;
            let cm  = state.op.cm;
            for (ri, &re) in res.iter().enumerate() {
                state.op.reinf = re;
                let cdf = skin_friction_drag(&state);
                let cd  = cdp + cdf;
                let pt = if cl.is_finite() && cd > 1e-9 {
                    Some(Point { alpha, cl, cd, cdp, cdf, cm, ld: cl / cd })
                } else { None };
                out[ri].push(pt);
            }
        }
        Ok(out)
    }

    // ── Batch: many airfoils at once ──────────────────────────────────────────

    /// Analyze many airfoils at the same α and Re in one batched GPU dispatch.
    ///
    /// Much faster than calling [`analyze`](Self::analyze) in a loop when you
    /// have 10+ candidates — the GPU processes all geometries in one command
    /// buffer.
    ///
    /// ```no_run
    /// # use titaniumfoil::Solver;
    /// let solver = Solver::new();
    /// let nacas = vec!["4412", "2412", "63412", "65415", "0012"];
    /// let results = solver.analyze_batch(&nacas, 4.0, 200_000.0);
    /// for (naca, r) in nacas.iter().zip(&results) {
    ///     if let Ok(Some(p)) = r { println!("{naca}: L/D={:.1}", p.ld); }
    /// }
    /// ```
    pub fn analyze_batch(
        &self,
        nacas:     &[&str],
        alpha_deg: f64,
        re:        f64,
    ) -> Vec<Result<Option<Point>, Error>> {
        use rayon::prelude::*;

        // Build all states in parallel
        let built: Vec<Result<XfoilState, Error>> = nacas.par_iter()
            .map(|naca| self.build_panel_state(naca, re))
            .collect();

        // Separate errors from valid states
        let mut states: Vec<XfoilState> = Vec::new();
        let mut indices: Vec<usize>      = Vec::new();
        let mut errors:  Vec<(usize, Error)> = Vec::new();
        for (i, b) in built.into_iter().enumerate() {
            match b {
                Ok(s)  => { states.push(s); indices.push(i); }
                Err(e) => errors.push((i, e)),
            }
        }

        // ONE batched GPU dispatch for all valid states (runs on calling thread)
        GPU.with(|ctx| {
            compute_panel_matrix_batch_gpu(&ctx.borrow(), &mut states);
        });

        // LU + alpha sweep in parallel
        let results: Vec<Option<Point>> = states.par_iter_mut().map(|state| {
            ggcalc_finish(state);
            state.op.alfa = alpha_deg * PI / 180.0;
            specal(state);
            Point::from_state(state, alpha_deg)
        }).collect();

        // Reassemble in original order
        let mut out: Vec<Result<Option<Point>, Error>> =
            (0..nacas.len()).map(|_| Ok(None)).collect();
        for (idx, pt) in indices.into_iter().zip(results) {
            out[idx] = Ok(pt);
        }
        for (idx, e) in errors {
            out[idx] = Err(e);
        }
        out
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn build_panel_state(&self, naca: &str, re: f64) -> Result<XfoilState, Error> {
        let mut state = XfoilState::default();
        load_naca_n(&mut state, naca, self.nside);
        if state.geom.n == 0 {
            return Err(Error::UnknownAirfoil(naca.to_string()));
        }
        state.op.qinf  = 1.0;
        state.op.reinf = re;
        state.op.minf  = 0.0;
        calc_normals(&mut state);
        calc_panel_angles(&mut state);
        Ok(state)
    }
}
