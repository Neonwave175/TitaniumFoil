use std::f64::consts::PI;
use titaniumfoil_core::types::XfoilState;
use titaniumfoil_core::geometry::{load_naca, load_naca_n};
use titaniumfoil_core::panel::{calc_normals, calc_panel_angles, specal, ggcalc_finish};
use titaniumfoil_core::viscous::skin_friction_drag;
use titaniumfoil_metal::{MetalContext, panel_gpu::compute_panel_matrix_gpu};

mod plot;
mod llt;

fn build_state(naca: &str, nside: usize, re: f64) -> XfoilState {
    let mut state = XfoilState::default();
    if nside == 65 { load_naca(&mut state, naca); }
    else           { load_naca_n(&mut state, naca, nside); }
    state.op.qinf  = 1.0;
    state.op.reinf = re;
    state.op.minf  = 0.0;
    calc_normals(&mut state);
    calc_panel_angles(&mut state);
    state
}

fn solve(state: &mut XfoilState, ctx: &MetalContext, alpha_deg: f64) {
    state.op.alfa = alpha_deg * PI / 180.0;
    compute_panel_matrix_gpu(ctx, state);
    ggcalc_finish(state);
    specal(state);
}

fn main() {
    let raw: Vec<String> = std::env::args().collect();

    // Flags: --no-plot / --plot (default on)
    let plot = !raw.iter().any(|a| a == "--no-plot");

    // Strip flags so positional parsing is unaffected
    let args: Vec<&String> = raw.iter()
        .filter(|a| !a.starts_with("--"))
        .collect();

    // titaniumfoil <naca> <alpha>                           — single point
    // titaniumfoil <naca> aseq <a1> <a2> <da> [re] [n]     — polar sweep
    // titaniumfoil <naca> wing <AR> [taper] [twist] [alpha] [re] [n]  — lift distribution
    let naca  = args.get(1).map(|s| s.as_str()).unwrap_or("0012");
    let mode  = args.get(2).map(|s| s.to_lowercase()).unwrap_or_default();
    let aseq  = mode == "aseq";
    let wing  = mode == "wing";

    if wing {
        let ar:    f64   = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6.0);
        let taper: f64   = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.5);
        let twist: f64   = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let alpha: f64   = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(4.0);
        let re:    f64   = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(200_000.0);
        let nside: usize = args.get(8).and_then(|s| s.parse().ok()).unwrap_or(65);

        let ctx = MetalContext::new();

        // ── Get 2-D lift-curve slope and zero-lift angle from panel solver ────
        let mut st = build_state(naca, nside, re);
        solve(&mut st, &ctx, 0.0);
        let cl0 = st.op.cl;

        let mut st5 = build_state(naca, nside, re);
        solve(&mut st5, &ctx, 5.0);
        let cl5  = st5.op.cl;
        let a0   = (cl5 - cl0) / (5.0_f64.to_radians());        // 1/rad
        let al0  = -cl0 / a0 * (180.0 / PI);                    // degrees

        // ── Solve LLT ─────────────────────────────────────────────────────────
        let res = llt::solve(ar, taper, twist, alpha, a0, al0);

        // ── Print header ───────────────────────────────────────────────────────
        println!();
        println!("  NACA {naca}   AR={ar:.1}   taper={taper:.2}   twist={twist:.1}°   α={alpha:.1}°   Re={re:.0}");
        println!("  a0={a0:.4}/rad   αL0={al0:.3}°");
        println!();
        println!("  CL_wing = {:.4}   CDi = {:.5}   e = {:.4}",
            res.cl_wing, res.cdi, res.span_eff);
        println!();
        println!("  {:>6}  {:>6}  {:>8}  {:>8}", "y/(b/2)", "c/cr", "CL_loc", "CL_ell");
        println!("  ──────  ──────  ────────  ────────");
        for i in (0..res.y_norm.len()).step_by(5) {
            println!("  {:>6.3}  {:>6.3}  {:>8.4}  {:>8.4}",
                res.y_norm[i], res.c_ratio[i], res.cl_local[i], res.cl_elliptic[i]);
        }
        println!();

        // ── Plot spanwise distribution ─────────────────────────────────────────
        let actual: Vec<(f64, f64)>   = res.y_norm.iter().zip(res.cl_local.iter())
            .map(|(&y, &cl)| (y, cl)).collect();
        let elliptic: Vec<(f64, f64)> = res.y_norm.iter().zip(res.cl_elliptic.iter())
            .map(|(&y, &cl)| (y, cl)).collect();

        if plot {
            plot::print_plot(
                &format!("spanwise lift distribution  NACA {naca}  AR={ar:.1}  α={alpha:.1}°"),
                "y/(b/2)  →  tip",
                "CL",
                &[
                    plot::Series { label: "actual",    color: "\x1b[38;5;117m", data: &actual   },
                    plot::Series { label: "elliptic",  color: "\x1b[38;5;240m", data: &elliptic },
                ],
                56, 20,
            );
        }

    } else if aseq {
        let a1:    f64   = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(-5.0);
        let a2:    f64   = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10.0);
        let da:    f64   = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let re:    f64   = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(1e6);
        let nside: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(65);

        let ctx = MetalContext::new();
        let mut state = build_state(naca, nside, re);

        // silent warm-up
        solve(&mut state, &ctx, a1);
        state = build_state(naca, nside, re);

        println!("  NACA {}   Re={:.0}   N={}", naca, re, state.geom.n);
        println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
            "α (°)", "CL", "CM", "CDP", "CDF", "L/D");
        println!("  ──────  ────────  ────────  ────────  ────────  ────────");

        let mut alphas = Vec::new();
        let mut cls    = Vec::new();
        let mut lds    = Vec::new();
        let mut cps    = Vec::new();

        let n_steps = ((a2 - a1) / da).round() as i64 + 1;
        for step in 0..n_steps {
            let alpha = a1 + step as f64 * da;
            if (da > 0.0 && alpha > a2 + 1e-9) || (da < 0.0 && alpha < a2 - 1e-9) { break; }
            solve(&mut state, &ctx, alpha);
            let cdp = state.op.cdp;
            let cdf = skin_friction_drag(&state);
            let cd  = cdp + cdf;
            let ld  = if cd > 1e-9 { state.op.cl / cd } else { 0.0 };
            println!("  {:>6.2}  {:>8.5}  {:>8.5}  {:>8.5}  {:>8.5}  {:>8.2}",
                alpha, state.op.cl, state.op.cm, cdp, cdf, ld);

            // Collect for graphs — Cp at ~25% chord (suction peak proxy, negated)
            let n  = state.geom.n;
            let i  = (n / 4).min(n - 1);
            alphas.push(alpha);
            cls.push(state.op.cl);
            lds.push(ld);
            cps.push(-state.vel.cpi[i]);
        }
        println!();
        if plot { plot::print_polar(naca, re, &alphas, &cls, &lds, &cps); }

    } else {
        let alpha: f64   = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5.0);
        let alpha = if wing { 5.0 } else { alpha }; // suppress unused-variable warning
        let re:    f64   = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1e6);
        let nside: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(65);

        let ctx = MetalContext::new();
        let mut state = build_state(naca, nside, re);

        // silent warm-up
        solve(&mut state, &ctx, alpha);
        state = build_state(naca, nside, re);

        solve(&mut state, &ctx, alpha);

        let n   = state.geom.n;
        let cdp = state.op.cdp;
        let cdf = skin_friction_drag(&state);
        let cd  = cdp + cdf;
        let ld  = if cd > 1e-9 { state.op.cl / cd } else { 0.0 };
        println!("  NACA {}   α={:.1}°   Re={:.0}   N={}", naca, alpha, re, n);
        println!("  CL={:.5}   CM={:.5}   CDP={:.5}   CDF={:.5}   CD={:.5}   L/D={:.2}",
            state.op.cl, state.op.cm, cdp, cdf, cd, ld);
        // ── Cp distribution plot ──────────────────────────────────────────────
        // Find leading-edge index (minimum x)
        let i_le = (0..n)
            .min_by(|&a, &b| state.geom.x[a].partial_cmp(&state.geom.x[b]).unwrap())
            .unwrap_or(n / 2);

        // Upper surface: LE → TE, plot −Cp (convention: suction up)
        let upper: Vec<(f64,f64)> = (0..=i_le).rev()
            .map(|i| (state.geom.x[i], -state.vel.cpi[i]))
            .collect();
        // Lower surface: LE → TE
        let lower: Vec<(f64,f64)> = (i_le..n)
            .map(|i| (state.geom.x[i], -state.vel.cpi[i]))
            .collect();

        if plot {
            plot::print_plot(
                &format!("−Cp distribution  NACA {}  α={:.1}°  Re={:.0}", naca, alpha, re),
                "x/c  →  trailing edge",
                "−Cp",
                &[
                    plot::Series { label: "upper", color: "\x1b[38;5;117m", data: &upper },
                    plot::Series { label: "lower", color: "\x1b[38;5;114m", data: &lower },
                ],
                56, 20,
            );
        }
    }
}
