//! PyO3 bindings for TitaniumFoil.
//!
//! Build & install into the current Python environment:
//!
//! ```bash
//! cd crates/titaniumfoil-py
//! pip install maturin
//! maturin develop --release
//! ```
//!
//! Then in Python:
//!
//! ```python
//! from titaniumfoil import Solver
//!
//! solver = Solver()
//! p = solver.analyze("4412", 4.0, 200_000)
//! print(f"CL={p.cl:.4f}  L/D={p.ld:.1f}")
//! ```

use pyo3::prelude::*;
use tf::{Solver as TfSolver, Point as TfPoint};

// ── Point ─────────────────────────────────────────────────────────────────────

/// Aerodynamic coefficients at one operating point.
///
/// Attributes
/// ----------
/// alpha : float   angle of attack (degrees)
/// cl    : float   lift coefficient
/// cd    : float   total drag  = cdp + cdf
/// cdp   : float   pressure drag
/// cdf   : float   skin-friction drag
/// cm    : float   pitching moment (quarter-chord)
/// ld    : float   lift-to-drag ratio
#[pyclass(get_all)]
#[derive(Clone)]
struct Point {
    alpha: f64,
    cl:    f64,
    cd:    f64,
    cdp:   f64,
    cdf:   f64,
    cm:    f64,
    ld:    f64,
}

#[pymethods]
impl Point {
    fn __repr__(&self) -> String {
        format!(
            "Point(α={:.1}°  CL={:.4}  CD={:.5}  CM={:.4}  L/D={:.1})",
            self.alpha, self.cl, self.cd, self.cm, self.ld
        )
    }

    /// Return as a plain dict for pandas/numpy interop.
    fn to_dict<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyDict> {
        use pyo3::types::IntoPyDict;
        [("alpha", self.alpha), ("cl", self.cl), ("cd", self.cd),
         ("cdp",  self.cdp),   ("cdf", self.cdf), ("cm", self.cm),
         ("ld",   self.ld)]
            .into_py_dict_bound(py)
    }
}

impl From<TfPoint> for Point {
    fn from(p: TfPoint) -> Self {
        Self { alpha: p.alpha, cl: p.cl, cd: p.cd, cdp: p.cdp,
               cdf: p.cdf, cm: p.cm, ld: p.ld }
    }
}

// ── Solver ────────────────────────────────────────────────────────────────────

/// GPU-accelerated NACA airfoil solver.
///
/// Parameters
/// ----------
/// nside : int, optional
///     Panel nodes per surface (total N = 2*nside - 1).
///     Default 65 (N=129, ~21 000 eval/s on M-series).
///
/// Examples
/// --------
/// >>> from titaniumfoil import Solver
/// >>> s = Solver()
/// >>> p = s.analyze("4412", 4.0, 200_000)
/// >>> print(p.ld)
///
/// >>> # Higher accuracy
/// >>> s = Solver(nside=120)
#[pyclass]
struct Solver {
    inner: TfSolver,
}

#[pymethods]
impl Solver {
    #[new]
    #[pyo3(signature = (nside=65))]
    fn new(nside: usize) -> Self {
        Self { inner: TfSolver::with_panels(nside) }
    }

    /// Compute one operating point.
    ///
    /// Parameters
    /// ----------
    /// naca      : str   NACA designation, e.g. "4412", "63412"
    /// alpha_deg : float angle of attack in degrees
    /// re        : float chord Reynolds number
    ///
    /// Returns
    /// -------
    /// Point | None   None if the solver diverges
    ///
    /// Examples
    /// --------
    /// >>> p = solver.analyze("4412", 4.0, 200_000)
    /// >>> print(p.cl, p.ld)
    fn analyze(
        &self, py: Python<'_>,
        naca: &str, alpha_deg: f64, re: f64,
    ) -> PyResult<Option<Point>> {
        let inner = &self.inner;
        let result = py.allow_threads(|| inner.analyze(naca, alpha_deg, re));
        result
            .map(|opt| opt.map(Point::from))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Compute a polar sweep — panel matrix built once, cheap per α.
    ///
    /// Parameters
    /// ----------
    /// naca   : str
    /// alphas : list[float]   angles of attack in degrees
    /// re     : float
    ///
    /// Returns
    /// -------
    /// list[Point | None]
    ///
    /// Examples
    /// --------
    /// >>> polar = solver.polar("4412", list(range(-5, 16)), 200_000)
    /// >>> best = max((p for p in polar if p), key=lambda p: p.ld)
    fn polar(
        &self, py: Python<'_>,
        naca: &str, alphas: Vec<f64>, re: f64,
    ) -> PyResult<Vec<Option<Point>>> {
        let inner = &self.inner;
        let result = py.allow_threads(|| inner.polar(naca, &alphas, re));
        result
            .map(|v| v.into_iter().map(|o| o.map(Point::from)).collect())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Polar across multiple Reynolds numbers.
    ///
    /// Parameters
    /// ----------
    /// naca   : str
    /// alphas : list[float]
    /// res    : list[float]
    ///
    /// Returns
    /// -------
    /// list[list[Point | None]]   indexed as [re_idx][alpha_idx]
    ///
    /// Examples
    /// --------
    /// >>> grid = solver.polar_multi_re("63412", [0,2,4,6,8], [80e3,150e3,300e3])
    /// >>> re80k_polar = grid[0]
    fn polar_multi_re(
        &self, py: Python<'_>,
        naca: &str, alphas: Vec<f64>, res: Vec<f64>,
    ) -> PyResult<Vec<Vec<Option<Point>>>> {
        let inner = &self.inner;
        let result = py.allow_threads(|| inner.polar_multi_re(naca, &alphas, &res));
        result
            .map(|grid| grid.into_iter()
                .map(|row| row.into_iter().map(|o| o.map(Point::from)).collect())
                .collect())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Evaluate many airfoils at one operating point in a batched GPU dispatch.
    ///
    /// Parameters
    /// ----------
    /// nacas     : list[str]
    /// alpha_deg : float
    /// re        : float
    ///
    /// Returns
    /// -------
    /// list[Point | None]
    ///
    /// Examples
    /// --------
    /// >>> results = solver.analyze_batch(["4412","2412","63412","65415"], 4.0, 200e3)
    /// >>> for naca, p in zip(nacas, results):
    /// ...     if p: print(f"{naca}: L/D={p.ld:.1f}")
    fn analyze_batch(
        &self, py: Python<'_>,
        nacas: Vec<String>, alpha_deg: f64, re: f64,
    ) -> Vec<Option<Point>> {
        let inner   = &self.inner;
        let refs: Vec<&str> = nacas.iter().map(|s| s.as_str()).collect();
        py.allow_threads(|| inner.analyze_batch(&refs, alpha_deg, re))
            .into_iter()
            .map(|r| r.ok().flatten().map(Point::from))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("Solver(nside={}, N={})", self.inner.nside, 2 * self.inner.nside - 1)
    }
}

// ── Module ────────────────────────────────────────────────────────────────────

#[pymodule]
fn titaniumfoil(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Solver>()?;
    m.add_class::<Point>()?;
    Ok(())
}
