#pragma once
// titaniumfoil.hpp — C++ wrapper around the TitaniumFoil Rust library.
//
// Link against:
//   macOS:  -L/path/to/target/release -ltitaniumfoil -framework Metal \
//           -framework Foundation -framework CoreGraphics
//
// Example:
//   #include "titaniumfoil.hpp"
//   tf::Solver solver;
//   auto p = solver.analyze("4412", 4.0, 200'000);
//   if (p) std::cout << p->ld << "\n";

#include <optional>
#include <string>
#include <vector>
#include <stdexcept>

// ── C declarations (must match crates/titaniumfoil-ffi/src/lib.rs) ────────────
extern "C" {

struct TfPoint {
    double alpha, cl, cd, cdp, cdf, cm, ld;
};

struct TfSolver;   // opaque
struct TfPolar;    // opaque

TfSolver* tf_solver_new(size_t nside);
void      tf_solver_free(TfSolver* solver);

int       tf_analyze(const TfSolver* solver, const char* naca,
                     double alpha_deg, double re, TfPoint* out);

TfPolar*  tf_polar(const TfSolver* solver, const char* naca,
                   const double* alphas, size_t n, double re);
size_t    tf_polar_len(const TfPolar* polar);
int       tf_polar_get(const TfPolar* polar, size_t index, TfPoint* out);
void      tf_polar_free(TfPolar* polar);

int       tf_polar_multi_re(const TfSolver* solver, const char* naca,
                             const double* alphas, size_t n_alpha,
                             const double* res,    size_t n_re,
                             TfPoint** out_pts, int** out_valid);
void      tf_free_points(TfPoint* ptr, size_t len);
void      tf_free_i32(int* ptr, size_t len);

} // extern "C"

// ── C++ wrapper ───────────────────────────────────────────────────────────────
namespace tf {

/// Aerodynamic coefficients at one operating point.
struct Point {
    double alpha; ///< angle of attack (degrees)
    double cl;    ///< lift coefficient
    double cd;    ///< total drag = cdp + cdf
    double cdp;   ///< pressure drag
    double cdf;   ///< skin-friction drag
    double cm;    ///< pitching moment (quarter-chord)
    double ld;    ///< lift-to-drag ratio
};

inline Point from_c(const TfPoint& p) {
    return { p.alpha, p.cl, p.cd, p.cdp, p.cdf, p.cm, p.ld };
}

/// GPU-accelerated NACA airfoil solver.
///
/// Move-only — copy is deleted because the underlying GPU context is not
/// safely copyable.
class Solver {
    TfSolver* _s;
public:
    /// @param nside  Panel nodes per surface (65 = default, 120 = better, 180 = best).
    explicit Solver(int nside = 65)
        : _s(tf_solver_new(static_cast<size_t>(nside)))
    {
        if (!_s) throw std::runtime_error("TitaniumFoil: failed to create Solver");
    }

    ~Solver() { tf_solver_free(_s); }

    Solver(const Solver&)            = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&& o) noexcept : _s(o._s) { o._s = nullptr; }

    // ── Single operating point ────────────────────────────────────────────────

    /// Compute CL, CD, CM, L/D at one angle of attack and Reynolds number.
    /// Returns std::nullopt if the NACA designation is invalid or the solve
    /// diverges (e.g. at extreme angles).
    ///
    /// @code
    /// tf::Solver s;
    /// if (auto p = s.analyze("4412", 4.0, 200'000)) {
    ///     std::cout << "L/D = " << p->ld << "\n";
    /// }
    /// @endcode
    std::optional<Point> analyze(const std::string& naca,
                                 double alpha_deg,
                                 double re) const
    {
        TfPoint out{};
        if (tf_analyze(_s, naca.c_str(), alpha_deg, re, &out))
            return from_c(out);
        return std::nullopt;
    }

    // ── Polar sweep ───────────────────────────────────────────────────────────

    /// Compute a full polar — panel matrix built once, cheap per alpha.
    ///
    /// @code
    /// std::vector<double> alphas;
    /// for (int a = -5; a <= 15; ++a) alphas.push_back(a);
    /// auto polar = solver.polar("4412", alphas, 200'000);
    /// for (auto& p : polar)
    ///     if (p) std::cout << p->alpha << "  " << p->ld << "\n";
    /// @endcode
    std::vector<std::optional<Point>> polar(const std::string&         naca,
                                            const std::vector<double>& alphas,
                                            double                     re) const
    {
        TfPolar* raw = tf_polar(_s, naca.c_str(), alphas.data(), alphas.size(), re);
        if (!raw) return {};
        size_t n = tf_polar_len(raw);
        std::vector<std::optional<Point>> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            TfPoint pt{};
            if (tf_polar_get(raw, i, &pt)) out.push_back(from_c(pt));
            else                           out.push_back(std::nullopt);
        }
        tf_polar_free(raw);
        return out;
    }

    // ── Multi-Re polar ────────────────────────────────────────────────────────

    /// Compute polars for multiple Reynolds numbers.
    /// Returns result[re_idx][alpha_idx].
    ///
    /// @code
    /// auto grid = solver.polar_multi_re("4412", {0,2,4,6,8}, {80e3,150e3,300e3});
    /// for (size_t ri = 0; ri < grid.size(); ++ri)
    ///     for (auto& p : grid[ri])
    ///         if (p) std::cout << p->ld << "\n";
    /// @endcode
    std::vector<std::vector<std::optional<Point>>>
    polar_multi_re(const std::string&         naca,
                   const std::vector<double>& alphas,
                   const std::vector<double>& res) const
    {
        TfPoint* pts   = nullptr;
        int*     valid = nullptr;
        size_t na = alphas.size(), nr = res.size();

        if (!tf_polar_multi_re(_s, naca.c_str(),
                               alphas.data(), na,
                               res.data(),   nr,
                               &pts, &valid))
            return {};

        std::vector<std::vector<std::optional<Point>>> out(nr);
        for (size_t ri = 0; ri < nr; ++ri) {
            out[ri].reserve(na);
            for (size_t ai = 0; ai < na; ++ai) {
                size_t idx = ri * na + ai;
                if (valid[idx]) out[ri].push_back(from_c(pts[idx]));
                else            out[ri].push_back(std::nullopt);
            }
        }
        tf_free_points(pts,   nr * na);
        tf_free_i32(valid, nr * na);
        return out;
    }
};

} // namespace tf
