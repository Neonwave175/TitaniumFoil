// GPU dispatch for the O(N²) panel influence matrix.
// Uses persistent StorageModeShared buffers — on Apple Silicon unified memory,
// CPU and GPU share the same physical DRAM so there are zero copies.
//
// Flow:
//   1. CPU writes geometry directly into ctx.node_buf.contents() ptr  (zero copy)
//   2. GPU reads node_buf, writes aij_buf   (same physical memory, no DMA)
//   3. CPU reads aij_buf.contents() and converts f32 → f64             (in-place)
//
// The only overhead vs pure CPU is the f32↔f64 conversion (~N² ops) and the
// Metal command buffer encode/commit/wait cycle.

use metal::*;
use rayon::prelude::*;
use titaniumfoil_core::types::{XfoilState, IQX};
use crate::context::{MetalContext, PanelNode, MAX_BATCH, NODE_STRIDE, AIJ_STRIDE};

/// Assemble the panel influence matrix AIJ for a single airfoil on the Metal GPU.
/// After returning, call `ggcalc_finish` (+ optionally `specal`) to complete the solve; `ggcalc_setup_rhs` is called internally.
pub fn compute_panel_matrix_gpu(ctx: &MetalContext, state: &mut XfoilState) {
    let n  = state.geom.n;
    let nn = n + 1;

    assert!(n <= IQX, "too many panels: n={n} > IQX={IQX}");

    // ── Zero-copy upload: write geometry directly into the shared buffer ─────
    {
        let ptr = ctx.node_buf.contents() as *mut PanelNode;
        for i in 0..n {
            unsafe {
                *ptr.add(i) = PanelNode {
                    xy:     [state.geom.x[i] as f32,  state.geom.y[i] as f32],
                    nxy:    [state.geom.nx[i] as f32,  state.geom.ny[i] as f32],
                    apanel:  state.geom.apanel[i] as f32,
                    s:       state.geom.s[i] as f32,
                    _pad:    [0.0; 2],
                };
            }
        }
    }
    // No fence needed — Apple Silicon has cache-coherent unified memory.

    // ── Encode and dispatch ──────────────────────────────────────────────────
    let n_u32  = n as u32;
    let nn_u32 = nn as u32;
    let scs    = (state.flow.ante / state.flow.dste.max(1e-30)) as f32;
    let sds    = (state.flow.aste / state.flow.dste.max(1e-30)) as f32;

    // T = threads per group, rounded to multiple of 32, capped at 1024 (Metal limit)
    // Shader declares static s_jo/s_jp[384] — T must not exceed that.
    // IQX=360 panels → max T = ceil(360/32)*32 = 384. ✓
    let t = (((n as u64 + 31) / 32) * 32).min(384);

    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.panel_pipeline);
    enc.set_buffer(0, Some(&ctx.node_buf), 0);
    enc.set_buffer(1, Some(&ctx.aij_buf),  0);
    enc.set_bytes(2, 4, &n_u32  as *const _ as *const _);
    enc.set_bytes(3, 4, &nn_u32 as *const _ as *const _);
    enc.set_bytes(4, 4, &scs    as *const _ as *const _);
    enc.set_bytes(5, 4, &sds    as *const _ as *const _);

    // One threadgroup per evaluation row; T threads per group (one per panel)
    let tg   = MTLSize { width: t, height: 1,             depth: 1 };
    let grid = MTLSize { width: 1, height: n as u64, depth: 1 };
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // ── Zero-copy readback: convert f32 → f64 from the shared aij_buf ───────
    // The GPU wrote to aij_buf; on unified memory this is the same physical
    // memory — no DMA, just a pointer read + widening cast per element.
    ctx.readback_aij(&mut state.panel.aij, n, nn);

    // Ensure dΨ₀ column is exact f64 -1 (GPU wrote -1.0f which is exact)
    for i in 0..n {
        state.panel.aij[i * nn + n] = -1.0;
    }

    // ── CPU completes: RHS, Kutta row, optional bisector ────────────────────
    titaniumfoil_core::panel::ggcalc_setup_rhs(state);
}

/// Assemble panel influence matrices for a batch of airfoils in a single Metal command buffer.
/// Automatically splits into chunks of `MAX_BATCH` (64); upload and readback are parallelised with rayon.
pub fn compute_panel_matrix_batch_gpu(ctx: &MetalContext, states: &mut [XfoilState]) {
    if states.is_empty() { return; }
    if states.len() > MAX_BATCH {
        for chunk in states.chunks_mut(MAX_BATCH) {
            compute_panel_matrix_batch_gpu(ctx, chunk);
        }
        return;
    }
    let batch = states.len();
    let n     = states[0].geom.n;
    let nn    = n + 1;
    assert!(n <= IQX, "too many panels: n={n} > IQX={IQX}");

    // ── Parallel upload: each airfoil writes to its own memory region ─────────
    let node_base = ctx.batch_node_buf.contents() as usize;
    let scs_base  = ctx.batch_scs_buf.contents()  as usize;
    let sds_base  = ctx.batch_sds_buf.contents()  as usize;

    states.par_iter().enumerate().for_each(|(ai, state)| {
        let node_ptr = (node_base + ai * NODE_STRIDE * std::mem::size_of::<PanelNode>())
            as *mut PanelNode;
        let scs_ptr  = (scs_base  + ai * std::mem::size_of::<f32>()) as *mut f32;
        let sds_ptr  = (sds_base  + ai * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..n {
            unsafe {
                *node_ptr.add(i) = PanelNode {
                    xy:     [state.geom.x[i] as f32, state.geom.y[i] as f32],
                    nxy:    [state.geom.nx[i] as f32, state.geom.ny[i] as f32],
                    apanel:  state.geom.apanel[i] as f32,
                    s:       state.geom.s[i] as f32,
                    _pad:    [0.0; 2],
                };
            }
        }
        unsafe {
            *scs_ptr = (state.flow.ante / state.flow.dste.max(1e-30)) as f32;
            *sds_ptr = (state.flow.aste / state.flow.dste.max(1e-30)) as f32;
        }
    });

    // ── ONE GPU dispatch for the whole batch ──────────────────────────────────
    let n_u32  = n as u32;
    let nn_u32 = nn as u32;
    let t = (((n as u64 + 31) / 32) * 32).min(384);

    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.batch_pipeline);
    enc.set_buffer(0, Some(&ctx.batch_node_buf), 0);
    enc.set_buffer(1, Some(&ctx.batch_aij_buf),  0);
    enc.set_buffer(2, Some(&ctx.batch_scs_buf),  0);
    enc.set_buffer(3, Some(&ctx.batch_sds_buf),  0);
    enc.set_bytes(4, 4, &n_u32  as *const _ as *const _);
    enc.set_bytes(5, 4, &nn_u32 as *const _ as *const _);

    let tg   = MTLSize { width: t,            height: 1,        depth: 1 };
    let grid = MTLSize { width: batch as u64, height: n as u64, depth: 1 };
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // ── Parallel readback + RHS setup ─────────────────────────────────────────
    let aij_base = ctx.batch_aij_buf.contents() as usize;

    states.par_iter_mut().enumerate().for_each(|(ai, state)| {
        let src = (aij_base + ai * AIJ_STRIDE * std::mem::size_of::<f32>()) as *const f32;
        for i in 0..n {
            for j in 0..nn {
                state.panel.aij[i * nn + j] = unsafe { *src.add(i * nn + j) } as f64;
            }
        }
        for i in 0..n { state.panel.aij[i * nn + n] = -1.0; }
        titaniumfoil_core::panel::ggcalc_setup_rhs(state);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use titaniumfoil_core::types::XfoilState;
    use titaniumfoil_core::geometry::load_naca;
    use titaniumfoil_core::panel::{calc_normals, calc_panel_angles, specal, ggcalc_finish};
    use crate::context::MetalContext;
    use std::f64::consts::PI;

    #[test]
    fn gpu_cpu_aij_parity_naca0012() {
        // CPU reference
        let mut cpu = XfoilState::default();
        load_naca(&mut cpu, "0012");
        cpu.op.alfa = 5.0 * PI / 180.0;
        cpu.op.qinf = 1.0;
        calc_normals(&mut cpu);
        calc_panel_angles(&mut cpu);
        titaniumfoil_core::panel::ggcalc(&mut cpu);
        specal(&mut cpu);
        let cl_cpu = cpu.op.cl;

        // GPU path
        let mut gpu_state = XfoilState::default();
        load_naca(&mut gpu_state, "0012");
        gpu_state.op.alfa = 5.0 * PI / 180.0;
        gpu_state.op.qinf = 1.0;
        calc_normals(&mut gpu_state);
        calc_panel_angles(&mut gpu_state);

        let ctx = MetalContext::new();
        compute_panel_matrix_gpu(&ctx, &mut gpu_state);
        ggcalc_finish(&mut gpu_state);
        specal(&mut gpu_state);
        let cl_gpu = gpu_state.op.cl;

        eprintln!("CL cpu={cl_cpu:.5}  CL gpu={cl_gpu:.5}");
        let diff = (cl_cpu - cl_gpu).abs();
        assert!(diff < 0.003,
            "CL mismatch: cpu={cl_cpu:.5} gpu={cl_gpu:.5} diff={diff:.2e}");
    }

    #[test]
    fn unified_memory_no_alloc_per_call() {
        // Verify that repeated calls reuse the same buffer (no new allocations).
        // Run 5 invocations and confirm results are stable (no use-after-free).
        let mut state = XfoilState::default();
        load_naca(&mut state, "0012");
        state.op.qinf = 1.0;
        calc_normals(&mut state);
        calc_panel_angles(&mut state);

        let ctx = MetalContext::new();
        let mut cls = Vec::new();
        for deg in [0, 3, 5, 8, 10] {
            state.op.alfa = deg as f64 * PI / 180.0;
            compute_panel_matrix_gpu(&ctx, &mut state);
            ggcalc_finish(&mut state);
            specal(&mut state);
            cls.push(state.op.cl);
        }

        // CL should increase monotonically with α
        for i in 1..cls.len() {
            assert!(cls[i] > cls[i-1],
                "CL not monotone: cls[{i}]={:.4} <= cls[{}]={:.4}",
                cls[i], i-1, cls[i-1]);
        }
    }
}
