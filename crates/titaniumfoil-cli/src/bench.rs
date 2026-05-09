use std::f64::consts::PI;
use titaniumfoil_core::types::XfoilState;
use titaniumfoil_core::geometry::{load_naca, load_naca_n};
use titaniumfoil_core::panel::{calc_normals, calc_panel_angles, specal, ggcalc, ggcalc_finish};
use titaniumfoil_metal::{MetalContext, panel_gpu::compute_panel_matrix_gpu};

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let naca  = args.get(1).map(|s| s.as_str()).unwrap_or("0012");
    let alpha: f64   = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let re:    f64   = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1e6);

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  TitaniumFoil benchmark  |  NACA {}  α={:.1}°  Re={:.0}", naca, alpha, re);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {:>8}  {:>10}  {:>10}  {:>8}", "N panels", "CPU (ms)", "GPU (ms)", "speedup");
    println!("  ────────  ──────────  ──────────  ────────");

    let ctx = MetalContext::new();

    // Panel sizes to benchmark
    let nsides = [65, 90, 121, 150, 180];

    for nside in nsides {
        let n = 2 * nside - 1;

        // CPU — single solve, 5 runs averaged
        let mut state = build_state(naca, nside, re);
        state.op.alfa = alpha * PI / 180.0;
        let cpu_ms = {
            let runs = 5;
            let t = std::time::Instant::now();
            for _ in 0..runs {
                ggcalc(&mut state);
                specal(&mut state);
            }
            t.elapsed().as_secs_f64() * 1000.0 / runs as f64
        };
        let cl_cpu = state.op.cl;

        // GPU warm-up (pipeline JIT, buffer priming)
        let mut gs = build_state(naca, nside, re);
        gs.op.alfa = alpha * PI / 180.0;
        compute_panel_matrix_gpu(&ctx, &mut gs);
        ggcalc_finish(&mut gs);
        specal(&mut gs);

        // GPU — 5 runs averaged
        let gpu_ms = {
            let runs = 5;
            let t = std::time::Instant::now();
            for _ in 0..runs {
                compute_panel_matrix_gpu(&ctx, &mut gs);
                ggcalc_finish(&mut gs);
                specal(&mut gs);
            }
            t.elapsed().as_secs_f64() * 1000.0 / runs as f64
        };
        let cl_gpu = gs.op.cl;
        let speedup = cpu_ms / gpu_ms.max(0.001);

        println!("  {:>8}  {:>10.2}  {:>10.2}  {:>7.1}×   ΔCL={:.1e}",
            n, cpu_ms, gpu_ms, speedup, (cl_cpu - cl_gpu).abs());
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Polar sweep benchmark at max N
    let nside = 180;
    let n_pts = 21i64;
    println!("\n  Polar sweep  α=-5→15° ({n_pts} pts)  N={}", 2*nside-1);
    println!("  ──────────────────────────────────────────────");

    let mut state = build_state(naca, nside, re);
    let t_cpu = std::time::Instant::now();
    for step in 0..n_pts {
        state.op.alfa = (-5.0 + step as f64) * PI / 180.0;
        ggcalc(&mut state);
        specal(&mut state);
    }
    let polar_cpu = t_cpu.elapsed().as_secs_f64() * 1000.0;

    let mut gs = build_state(naca, nside, re);
    // warm-up
    gs.op.alfa = 0.0;
    compute_panel_matrix_gpu(&ctx, &mut gs); ggcalc_finish(&mut gs);

    let t_gpu = std::time::Instant::now();
    for step in 0..n_pts {
        gs.op.alfa = (-5.0 + step as f64) * PI / 180.0;
        compute_panel_matrix_gpu(&ctx, &mut gs);
        ggcalc_finish(&mut gs);
        specal(&mut gs);
    }
    let polar_gpu = t_gpu.elapsed().as_secs_f64() * 1000.0;

    println!("  CPU : {polar_cpu:.1} ms total  ({:.1} ms/pt)", polar_cpu / n_pts as f64);
    println!("  GPU : {polar_gpu:.1} ms total  ({:.1} ms/pt)  →  {:.1}× faster",
        polar_gpu / n_pts as f64,
        polar_cpu / polar_gpu.max(0.001));
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
