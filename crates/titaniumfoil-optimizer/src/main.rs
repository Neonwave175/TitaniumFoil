// titaniumfoil-optimizer — NACA airfoil optimizer (4-digit, 6-series, or mixed)
//
// Rust port of OptimAerofoilmake.py (Neonwave175).
// Differential evolution over L/D = CL / (Cdp + Cdf) averaged across Reynolds numbers.

use std::f64::consts::PI;
use std::io::{self, Write};
use std::time::Instant;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use rayon::prelude::*;

use titaniumfoil_core::types::XfoilState;
use titaniumfoil_core::geometry::load_naca_n;
use titaniumfoil_core::panel::{calc_normals, calc_panel_angles, ggcalc_finish, specal};
use titaniumfoil_core::viscous::skin_friction_drag;
use titaniumfoil_metal::context::MetalContext;
use titaniumfoil_metal::panel_gpu::compute_panel_matrix_batch_gpu;

static BATCH_GPU: OnceLock<MetalContext> = OnceLock::new();
fn batch_gpu() -> &'static MetalContext {
    BATCH_GPU.get_or_init(MetalContext::new)
}

// Cache: designation string → (per-Re scores, avg).
// Within one run the config is fixed, so the same airfoil always gives the
// same result — no invalidation needed.  Two threads may race on the same
// designation and both compute it; that's fine, the result is idempotent.
static CACHE: OnceLock<Mutex<HashMap<String, (Vec<f64>, f64)>>> = OnceLock::new();
fn cache() -> &'static Mutex<HashMap<String, (Vec<f64>, f64)>> {
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

// ── ANSI colours ─────────────────────────────────────────────────────────────
const WHITE:  &str = "\x1b[38;5;253m";
const CYAN:   &str = "\x1b[38;5;117m";
const GREEN:  &str = "\x1b[38;5;114m";
const RED:    &str = "\x1b[38;5;203m";
const DIM:    &str = "\x1b[38;5;240m";
const MUTED:  &str = "\x1b[38;5;245m";
const RESET:  &str = "\x1b[0m";
const BOLD:   &str = "\x1b[1m";

// ── Airfoil family ────────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq)]
enum AirfoilKind { Naca4, Naca6, Mixed }

impl AirfoilKind {
    /// Number of DE dimensions.
    fn n_dims(&self) -> usize {
        match self { AirfoilKind::Mixed => 4, _ => 3 }
    }

    /// Parameter bounds for DE.  Mixed always uses fixed wide bounds.
    fn bounds(&self, cfg: &Config) -> Vec<(f64, f64)> {
        match self {
            AirfoilKind::Mixed => vec![
                (0.0,  1.0),   // family selector: <0.5 = 4-digit, ≥0.5 = 6-series
                (0.0,  6.0),   // p1: m max 6% (4-digit) / maps to series 63-65 (6-series)
                (0.0,  6.0),   // p2: p max 60% chord (4-digit) / cli digit 0-6 (6-series)
                (6.0, 18.0),   // p3: thickness (both families)
            ],
            _ => vec![
                (cfg.p1_min, cfg.p1_max),
                (cfg.p2_min, cfg.p2_max),
                (cfg.p3_min, cfg.p3_max),
            ],
        }
    }

    /// Build XFOIL designation from a parameter slice.
    fn designation(&self, p: &[f64]) -> String {
        match self {
            AirfoilKind::Naca4 => format!(
                "{}{}{:02}",
                p[0].round() as u32, p[1].round() as u32, p[2].round() as u32
            ),
            AirfoilKind::Naca6 => {
                let sd = (p[0].round() as u32).clamp(3, 7);
                let cd = (p[1].round() as u32).clamp(0, 9);
                let tp = (p[2].round() as u32).clamp(6, 21);
                format!("6{}{}{:02}", sd, cd, tp)
            }
            AirfoilKind::Mixed => {
                if p[0] < 0.5 {
                    // 4-digit: p[1]=m, p[2]=p, p[3]=t
                    format!("{}{}{:02}",
                        p[1].round() as u32, p[2].round() as u32, p[3].round() as u32)
                } else {
                    // 6-series: p[1] in [0,9] → series digit [3,7]
                    let sd = (3.0 + (p[1] / 9.0 * 4.0)).round().clamp(3.0, 7.0) as u32;
                    let cd = p[2].round().clamp(0.0, 9.0) as u32;
                    let tp = p[3].round().clamp(6.0, 21.0) as u32;
                    format!("6{}{}{:02}", sd, cd, tp)
                }
            }
        }
    }

    /// Display name (with formatting for 6-series hyphen).
    fn display_name(&self, p: &[f64]) -> String {
        match self {
            AirfoilKind::Naca4 => format!("NACA {}", self.designation(p)),
            AirfoilKind::Naca6 => {
                let sd = (p[0].round() as u32).clamp(3, 7);
                let cd = (p[1].round() as u32).clamp(0, 9);
                let tp = (p[2].round() as u32).clamp(6, 21);
                format!("NACA 6{}-{}{:02}", sd, cd, tp)
            }
            AirfoilKind::Mixed => {
                let desig = self.designation(p);
                if p[0] < 0.5 {
                    format!("[4] NACA {}", desig)
                } else {
                    format!("[6] NACA {}-{}", &desig[..2], &desig[2..])
                }
            }
        }
    }

    /// Parameter description line for the final result.
    fn param_line(&self, p: &[f64]) -> String {
        match self {
            AirfoilKind::Naca4 =>
                format!("m={:.3}  p={:.3}  t={:.3}", p[0], p[1], p[2]),
            AirfoilKind::Naca6 =>
                format!("series=6{:.0}  cli=0.{:.0}  t={:.3}%", p[0], p[1], p[2]),
            AirfoilKind::Mixed => {
                if p[0] < 0.5 {
                    format!("family=4-digit  m={:.3}  p={:.3}  t={:.3}", p[1], p[2], p[3])
                } else {
                    let sd = (3.0 + (p[1] / 9.0 * 4.0)).round().clamp(3.0, 7.0);
                    format!("family=6-series  series=6{:.0}  cli=0.{:.0}  t={:.3}%", sd, p[2], p[3])
                }
            }
        }
    }
}

// ── User configuration ────────────────────────────────────────────────────────
#[derive(Clone)]
struct Config {
    kind:           AirfoilKind,
    reynolds_range: Vec<f64>,
    aoa_start: f64,
    aoa_end:   f64,
    aoa_step:  f64,
    p1_min: f64, p1_max: f64,   // ignored in Mixed mode
    p2_min: f64, p2_max: f64,
    p3_min: f64, p3_max: f64,
    maxiter: usize,
    popsize: usize,
    seed:    u64,
    max_ld:  f64,
    nside:   usize,
    tol:     f64,
    cm_penalty: f64,  // subtract cm_penalty × |CM| from each alpha's L/D
}

// ── Terminal helpers ──────────────────────────────────────────────────────────
fn prompt(msg: &str) -> String {
    print!("{}", msg);
    io::stdout().flush().unwrap();
    let mut buf = String::new();
    io::stdin().read_line(&mut buf).unwrap();
    buf.trim().to_string()
}
fn parse_f(s: &str, d: f64)    -> f64    { if s.is_empty() { d } else { s.parse().unwrap_or(d) } }
fn parse_i(s: &str, d: usize)  -> usize  { if s.is_empty() { d } else { s.parse().unwrap_or(d) } }
fn parse_u(s: &str, d: u64)    -> u64    { if s.is_empty() { d } else { s.parse().unwrap_or(d) } }

fn get_user_parameters() -> Config {
    let bar = format!("{DIM}{}{RESET}", "─".repeat(60));
    println!("\n{bar}");
    println!("  {WHITE}{BOLD}TitaniumFoil naca airfoil optimizer{RESET}");
    println!("{bar}\n");

    // ── [0] Family ────────────────────────────────────────────────────────────
    println!("{MUTED}[0] airfoil family{RESET}");
    println!("    {DIM}4  →  NACA 4-digit  (e.g. 4412){RESET}");
    println!("    {DIM}6  →  NACA 6-series (e.g. 63-412){RESET}");
    println!("    {DIM}m  →  mixed 4+6     (search both simultaneously){RESET}");
    let kind_raw = prompt(&format!("    {DIM}[4/6/m]  (default 4)  >{RESET} "));
    let kind = match kind_raw.trim() {
        "6"             => AirfoilKind::Naca6,
        "m" | "M"       => AirfoilKind::Mixed,
        _               => AirfoilKind::Naca4,
    };
    let kind_name = match kind {
        AirfoilKind::Naca4  => "NACA 4-digit",
        AirfoilKind::Naca6  => "NACA 6-series",
        AirfoilKind::Mixed  => "mixed 4+6",
    };
    println!("    {DIM}using: {kind_name}{RESET}\n");

    // ── [1] Reynolds numbers ──────────────────────────────────────────────────
    println!("{MUTED}[1] reynolds numbers{RESET}  {DIM}default: 80000, 150000, 300000{RESET}");
    let raw = prompt(&format!("    {DIM}>{RESET} "));
    let reynolds_range: Vec<f64> = if raw.is_empty() {
        vec![80000.0, 150000.0, 300000.0]
    } else {
        raw.split(',').filter_map(|s| s.trim().parse().ok()).collect()
    };
    println!("    {DIM}using: {:?}{RESET}\n", reynolds_range);

    // ── [2] AoA sweep ─────────────────────────────────────────────────────────
    println!("{MUTED}[2] angle of attack sweep{RESET}");
    let aoa_start = parse_f(&prompt(&format!("    {DIM}start (default -5)  >{RESET} ")), -5.0);
    let aoa_end   = parse_f(&prompt(&format!("    {DIM}end   (default  5)  >{RESET} ")),  5.0);
    let aoa_step  = parse_f(&prompt(&format!("    {DIM}step  (default  2)  >{RESET} ")),  2.0);
    println!("    {DIM}sweep: {aoa_start} to {aoa_end}, step {aoa_step}{RESET}\n");

    // ── [3] Search bounds — skipped for Mixed (fixed wide bounds used) ────────
    let (p1_min, p1_max, p2_min, p2_max, p3_min, p3_max);
    if kind == AirfoilKind::Mixed {
        println!("{MUTED}[3] search bounds{RESET}  \
                  {DIM}mixed mode uses fixed bounds: family[0-1]  p1[0-9]  p2[0-9]  t[6-21]{RESET}\n");
        (p1_min, p1_max) = (0.0, 9.0);
        (p2_min, p2_max) = (0.0, 9.0);
        (p3_min, p3_max) = (6.0, 21.0);
    } else {
        let labels   = match kind {
            AirfoilKind::Naca4 => ["m (max-camber 0-9)", "p (camber-pos 1-9)", "t (thickness 6-18)"],
            _                  => ["series (3-7: 63–67)", "cli (0-9 → 0.0–0.9)", "t% (thickness 6-21)"],
        };
        let defaults: [(f64, f64); 3] = match kind {
            AirfoilKind::Naca4 => [(0.0, 6.0), (1.0, 6.0), (6.0, 18.0)],
            _                  => [(3.0, 7.0), (0.0, 9.0), (6.0, 21.0)],
        };
        println!("{MUTED}[3] search bounds{RESET}");
        println!("    {DIM}param 1: {}{RESET}", labels[0]);
        p1_min = parse_f(&prompt(&format!("      {DIM}min (default {:.0})  >{RESET} ", defaults[0].0)), defaults[0].0);
        p1_max = parse_f(&prompt(&format!("      {DIM}max (default {:.0})  >{RESET} ", defaults[0].1)), defaults[0].1);
        println!("    {DIM}param 2: {}{RESET}", labels[1]);
        p2_min = parse_f(&prompt(&format!("      {DIM}min (default {:.0})  >{RESET} ", defaults[1].0)), defaults[1].0);
        p2_max = parse_f(&prompt(&format!("      {DIM}max (default {:.0})  >{RESET} ", defaults[1].1)), defaults[1].1);
        println!("    {DIM}param 3: {}{RESET}", labels[2]);
        p3_min = parse_f(&prompt(&format!("      {DIM}min (default {:.0})  >{RESET} ", defaults[2].0)), defaults[2].0);
        p3_max = parse_f(&prompt(&format!("      {DIM}max (default {:.0})  >{RESET} ", defaults[2].1)), defaults[2].1);
        println!();
    }

    // ── [4] Optimizer ─────────────────────────────────────────────────────────
    println!("{MUTED}[4] optimizer{RESET}");
    let maxiter = parse_i(&prompt(&format!("    {DIM}max iterations (default 100) >{RESET} ")), 100);
    let popsize = parse_i(&prompt(&format!("    {DIM}population size (default  5) >{RESET} ")),   5);
    let seed    = parse_u(&prompt(&format!("    {DIM}random seed     (default 42) >{RESET} ")),  42);
    println!();

    println!("{MUTED}[5] l/d cap{RESET}  {DIM}values above this are convergence artifacts{RESET}");
    let max_ld = parse_f(&prompt(&format!("    {DIM}max l/d (default 60) >{RESET} ")), 60.0);

    println!("{MUTED}[6] panel resolution{RESET}  {DIM}nside per surface (default 65 → N=129){RESET}");
    let nside = parse_i(&prompt(&format!("    {DIM}nside (default 65) >{RESET} ")), 65);

    println!("{MUTED}[7] convergence tolerance{RESET}  \
              {DIM}stop when std/|mean| ≤ tol  (scipy default: 0.01, 0=run all){RESET}");
    let tol = parse_f(&prompt(&format!("    {DIM}tol (default 0.01) >{RESET} ")), 0.01);

    println!("{MUTED}[8] pitching moment penalty{RESET}  \
              {DIM}score = L/D − w×|CM|  penalises extreme camber/aft-loading{RESET}");
    println!("    {DIM}NACA 4412 CM≈−0.10 → penalty≈1.0   NACA 9806 CM≈−0.28 → penalty≈5.6{RESET}");
    let cm_penalty = parse_f(&prompt(&format!("    {DIM}w (default 20, 0=off) >{RESET} ")), 20.0);

    println!("\n{bar}\n");
    Config {
        kind, reynolds_range, aoa_start, aoa_end, aoa_step,
        p1_min, p1_max, p2_min, p2_max, p3_min, p3_max,
        maxiter, popsize, seed, max_ld, nside, tol, cm_penalty,
    }
}

// ── L/D evaluation ───────────────────────────────────────────────────────────

/// Build geometry + GPU panel matrix ONCE for an airfoil, then clone the
/// solved state for each Re value and run the α sweep in parallel.
///
/// Returns `(per_Re_scores, avg)` where avg treats failed Re as 0.
fn evaluate_batch(all_p: &[Vec<f64>], cfg: &Config) -> Vec<(Vec<f64>, f64)> {
    let designations: Vec<String> = all_p.iter().map(|p| cfg.kind.designation(p)).collect();

    let mut results: Vec<Option<(Vec<f64>, f64)>> = vec![None; all_p.len()];
    let mut miss_idx: Vec<usize> = Vec::new();
    if let Ok(c) = cache().lock() {
        for (i, d) in designations.iter().enumerate() {
            if let Some(hit) = c.get(d) { results[i] = Some(hit.clone()); }
            else                         { miss_idx.push(i); }
        }
    } else { miss_idx = (0..all_p.len()).collect(); }

    if !miss_idx.is_empty() {
        let mut bases: Vec<XfoilState> = miss_idx.par_iter().map(|&i| {
            let mut s = XfoilState::default();
            load_naca_n(&mut s, &designations[i], cfg.nside);
            s.op.qinf = 1.0; s.op.minf = 0.0;
            calc_normals(&mut s); calc_panel_angles(&mut s);
            s
        }).collect();

        compute_panel_matrix_batch_gpu(batch_gpu(), &mut bases);

        let miss_results: Vec<(Vec<f64>, f64)> = bases.par_iter_mut().map(|base| {
            ggcalc_finish(base);
            alpha_sweep(base, cfg)
        }).collect();

        if let Ok(mut c) = cache().lock() {
            for (&mi, res) in miss_idx.iter().zip(miss_results.iter()) {
                c.entry(designations[mi].clone()).or_insert_with(|| res.clone());
                results[mi] = Some(res.clone());
            }
        }
    }
    results.into_iter().map(|r| r.unwrap()).collect()
}

/// Alpha sweep on a state with its panel matrix already solved.
/// Returns (per-Re scores, avg) — fail→0 for avg.
fn alpha_sweep(base: &mut XfoilState, cfg: &Config) -> (Vec<f64>, f64) {
    let n_re    = cfg.reynolds_range.len();
    let n_steps = ((cfg.aoa_end - cfg.aoa_start) / cfg.aoa_step).round() as usize + 1;
    let mut ld_sums = vec![0.0f64; n_re];
    let mut counts  = vec![0usize;  n_re];

    for step in 0..n_steps {
        let alpha = cfg.aoa_start + step as f64 * cfg.aoa_step;
        if alpha > cfg.aoa_end + 1e-9 { break; }
        base.op.alfa = alpha * PI / 180.0;
        specal(base);
        let cl = base.op.cl; let cdp = base.op.cdp; let cm = base.op.cm;
        if !cl.is_finite() || !cdp.is_finite() || !cm.is_finite() { continue; }
        for (ri, &re) in cfg.reynolds_range.iter().enumerate() {
            base.op.reinf = re;
            let cd = cdp + skin_friction_drag(base);
            if cd > 1e-6 {
                let s = cl / cd - cfg.cm_penalty * cm.abs();
                if s.is_finite() && s <= cfg.max_ld { ld_sums[ri] += s; counts[ri] += 1; }
            }
        }
    }
    let scores: Vec<f64> = (0..n_re)
        .map(|i| if counts[i] == 0 { f64::NAN } else { ld_sums[i] / counts[i] as f64 })
        .collect();
    let avg = scores.iter().map(|&v| if v.is_finite() { v } else { 0.0 }).sum::<f64>()
        / n_re as f64;
    (scores, avg)
}

// ── Differential evolution (DE/rand/1/bin) ────────────────────────────────────
struct LcgRng(u64);
impl LcgRng {
    fn new(seed: u64) -> Self { Self(seed ^ 6364136223846793005) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn next_f64(&mut self)        -> f64   { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    fn next_usize(&mut self, n: usize) -> usize { (self.next_u64() % n as u64) as usize }
}

struct DeOptimizer {
    pop:       Vec<Vec<f64>>,
    scores:    Vec<f64>,
    re_scores: Vec<Vec<f64>>,
    bounds:    Vec<(f64, f64)>,
    n_dims:    usize,
    f:   f64,
    cr:  f64,
    rng: LcgRng,
    best_idx: usize,
}

impl DeOptimizer {
    fn new(cfg: &Config) -> Self {
        let bounds   = cfg.kind.bounds(cfg);
        let n_dims   = bounds.len();
        let pop_size = cfg.popsize * n_dims;
        let mut rng  = LcgRng::new(cfg.seed);

        let pop: Vec<Vec<f64>> = (0..pop_size).map(|_| {
            bounds.iter().map(|&(lo, hi)| lo + rng.next_f64() * (hi - lo)).collect()
        }).collect();

        let n_re      = cfg.reynolds_range.len();
        let scores    = vec![f64::NAN; pop_size];
        let re_scores = vec![vec![f64::NAN; n_re]; pop_size];
        Self { pop, scores, re_scores, bounds, n_dims, f: 0.8, cr: 0.7, rng, best_idx: 0 }
    }

    fn update_best(&mut self) {
        if let Some((i, _)) = self.scores.iter().enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        { self.best_idx = i; }
    }

    fn print_candidate(
        p: &[f64], re_sc: &[f64], avg: f64,
        eval_count: usize, elapsed: f64,
        cfg: &Config,
    ) {
        let name    = cfg.kind.display_name(p);
        let re_cols: Vec<String> = cfg.reynolds_range.iter().zip(re_sc.iter())
            .map(|(re, &s)| fmt_re_col(*re, s)).collect();
        let avg_color = if avg > 40.0 { GREEN } else { WHITE };
        let avg_str   = if avg > 0.0 { format!("{avg:.4}") } else { "fail".to_string() };
        let t_str     = fmt_elapsed(elapsed);
        println!(
            "  {DIM}#{:<4} [{t_str}]{RESET}  {WHITE}{BOLD}{name}{RESET}  {}  \
             {DIM}avg{RESET} {avg_color}{avg_str}{RESET}",
            eval_count, re_cols.join("  "),
        );
    }

    fn initialise(&mut self, cfg: &Config, eval_count: &mut usize, t0: Instant) {
        let results = evaluate_batch(&self.pop, cfg);
        for (i, p) in self.pop.iter().enumerate() {
            let (re_sc, avg) = &results[i];
            *eval_count += 1;
            Self::print_candidate(p, re_sc, *avg, *eval_count, t0.elapsed().as_secs_f64(), cfg);
            self.re_scores[i] = re_sc.clone();
            self.scores[i]    = *avg;
        }
        self.update_best();
    }

    fn generate_trials(&mut self) -> Vec<Vec<f64>> {
        let n = self.pop.len();
        let d = self.n_dims;
        let mut trials = Vec::with_capacity(n);
        for i in 0..n {
            let mut r = [0usize; 3];
            let mut k = 0;
            while k < 3 {
                let idx = self.rng.next_usize(n);
                if idx != i && !r[..k].contains(&idx) { r[k] = idx; k += 1; }
            }
            let [r1, r2, r3] = r;
            let j_rand = self.rng.next_usize(d);
            let mut trial = self.pop[i].clone();
            for dim in 0..d {
                if dim == j_rand || self.rng.next_f64() < self.cr {
                    let (lo, hi) = self.bounds[dim];
                    trial[dim] = (self.pop[r1][dim]
                        + self.f * (self.pop[r2][dim] - self.pop[r3][dim]))
                        .clamp(lo, hi);
                }
            }
            trials.push(trial);
        }
        trials
    }

    /// One DE generation.  Returns `true` when converged.
    fn step(&mut self, cfg: &Config, gen: usize, t0: Instant, eval_count: &mut usize) -> bool {
        let trials = self.generate_trials();
        let trial_results = evaluate_batch(&trials, cfg);

        for (i, p) in trials.iter().enumerate() {
            let (re_sc, avg) = &trial_results[i];
            *eval_count += 1;
            Self::print_candidate(p, re_sc, *avg, *eval_count, t0.elapsed().as_secs_f64(), cfg);
            if *avg > self.scores[i] {
                self.pop[i]       = p.clone();
                self.re_scores[i] = re_sc.clone();
                self.scores[i]    = *avg;
            }
        }
        self.update_best();

        // ── Generation summary + convergence ──────────────────────────────────
        let best_name = cfg.kind.display_name(&self.pop[self.best_idx]);
        let elapsed   = t0.elapsed().as_secs_f64();

        let finite: Vec<f64> = self.scores.iter().cloned().filter(|v| v.is_finite()).collect();
        let (conv_std, conv_mean) = if finite.len() > 1 {
            let mean = finite.iter().sum::<f64>() / finite.len() as f64;
            let std  = (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / finite.len() as f64).sqrt();
            (std, mean)
        } else { (f64::INFINITY, 1.0) };

        let ratio = conv_std / conv_mean.abs().max(1e-30);
        let t_str = fmt_elapsed(elapsed);
        let eps   = if elapsed > 0.0 { *eval_count as f64 / elapsed } else { 0.0 };
        println!(
            "\n  {DIM}gen {:<3} [{t_str}]  {:.0} eval/s{RESET}  \
             {WHITE}{BOLD}best {best_name}{RESET}  \
             {DIM}std/mean{RESET} {MUTED}{ratio:.6}{RESET}\n",
            gen + 1, eps
        );

        cfg.tol > 0.0 && conv_std <= cfg.tol * conv_mean.abs()
    }
}

// ── Formatting ────────────────────────────────────────────────────────────────
/// Format elapsed seconds as "1.23ms", "456ms", or "12.3s" depending on scale.
fn fmt_elapsed(secs: f64) -> String {
    let ms = secs * 1000.0;
    if ms < 1000.0 { format!("{ms:6.1}ms") } else { format!("{secs:6.2}s ") }
}

fn fmt_re_col(re: f64, score: f64) -> String {
    let label = format!("{CYAN}Re={}k{RESET}", (re / 1000.0) as u32);
    if score.is_finite() && score > 0.0 {
        let c = if score > 40.0 { GREEN } else { WHITE };
        format!("{label}{DIM}:{RESET}{c}{score:7.2}{RESET}")
    } else {
        format!("{label}{DIM}:{RESET}{RED}   fail{RESET}")
    }
}

fn print_sanity(re: f64, score: f64) {
    if score.is_finite() && score > 0.0 {
        println!("  {MUTED}Re={:.0}:{RESET} {WHITE}{score:.4}{RESET}", re);
    } else {
        println!("  {MUTED}Re={:.0}:{RESET} {RED}fail{RESET}", re);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
fn main() {
    let cfg = get_user_parameters();

    // Sanity check
    let (s_p, s_name): (Vec<f64>, &str) = match cfg.kind {
        AirfoilKind::Naca4  => (vec![4.0, 4.0, 12.0],        "NACA 4412"),
        AirfoilKind::Naca6  => (vec![3.0, 4.0, 12.0],        "NACA 63-412"),
        AirfoilKind::Mixed  => (vec![0.0, 4.0, 4.0, 12.0],   "NACA 4412 (4-digit side)"),
    };
    println!("{DIM}sanity check ({s_name}){RESET}");
    let results = evaluate_batch(&[s_p], &cfg);
    let (sanity_scores, _) = results.into_iter().next().unwrap();
    for (&re, &sc) in cfg.reynolds_range.iter().zip(sanity_scores.iter()) {
        print_sanity(re, sc);
    }
    println!();

    let n_workers = rayon::current_num_threads();
    let n_dims    = cfg.kind.n_dims();
    println!("{DIM}using {n_workers} rayon threads  ·  {n_dims} DE dimensions{RESET}\n");

    let t0 = Instant::now();
    let mut eval_count = 0usize;
    let mut de = DeOptimizer::new(&cfg);

    println!("{DIM}── initial population ──────────────────────────────────────{RESET}");
    de.initialise(&cfg, &mut eval_count, t0);
    println!("{DIM}── evolving ─────────────────────────────────────────────────{RESET}\n");

    for gen in 0..cfg.maxiter {
        let converged = de.step(&cfg, gen, t0, &mut eval_count);
        if converged {
            println!("{DIM}converged at generation {} (std/mean ≤ {:.4}){RESET}\n",
                gen + 1, cfg.tol);
            break;
        }
    }

    // ── Final result ──────────────────────────────────────────────────────────
    let best_p      = &de.pop[de.best_idx];
    let best_scores = &de.re_scores[de.best_idx];
    let best_ld     = de.scores[de.best_idx];
    let best_name   = cfg.kind.display_name(best_p);
    let best_desig  = cfg.kind.designation(best_p);
    let best_params = cfg.kind.param_line(best_p);

    let bar     = format!("{DIM}{}{RESET}", "─".repeat(60));
    let re_cols: Vec<String> = cfg.reynolds_range.iter().zip(best_scores.iter())
        .map(|(re, &s)| fmt_re_col(*re, s)).collect();
    let elapsed = t0.elapsed().as_secs_f64();

    println!("{bar}");
    println!("  {MUTED}airfoil{RESET}      {WHITE}{BOLD}{best_name}{RESET}");
    println!("  {MUTED}designation{RESET}  {DIM}{best_desig}{RESET}");
    println!("  {MUTED}params{RESET}       {DIM}{best_params}{RESET}");
    println!("  {MUTED}l/d per Re{RESET}   {}", re_cols.join("  "));
    println!("  {MUTED}avg l/d{RESET}      {GREEN}{best_ld:.4}{RESET}");
    let eps        = if elapsed > 0.0 { eval_count as f64 / elapsed } else { 0.0 };
    let cache_size = cache().lock().map(|c| c.len()).unwrap_or(0);
    let cache_hits = eval_count.saturating_sub(cache_size);
    println!("  {MUTED}evaluations{RESET}  {WHITE}{eval_count}  {DIM}({eps:.0} eval/s){RESET}");
    println!("  {MUTED}cache{RESET}        {DIM}{cache_size} unique  {cache_hits} hits ({:.0}%){RESET}",
        cache_hits as f64 / eval_count.max(1) as f64 * 100.0);
    println!("  {MUTED}time{RESET}         {WHITE}{}{RESET}", fmt_elapsed(elapsed));
    println!("{bar}\n");
}
