use metal::{Device, CommandQueue, ComputePipelineState, Library, Buffer};
use metal::MTLResourceOptions;
use titaniumfoil_core::types::IQX;

// Maximum number of airfoils in one batched dispatch.
pub const MAX_BATCH: usize = 64;

// Fixed per-airfoil strides in the batch buffers (must match shader constants).
pub const NODE_STRIDE: usize = IQX;                    // PanelNode slots
pub const AIJ_STRIDE:  usize = IQX * (IQX + 1);       // f32 slots = 129 960

#[repr(C)]
pub struct PanelNode {
    pub xy:     [f32; 2],
    pub nxy:    [f32; 2],
    pub apanel: f32,
    pub s:      f32,
    pub _pad:   [f32; 2],
}

/// Persistent Metal GPU context holding pre-allocated shared buffers for panel matrix assembly.
///
/// Create once with `MetalContext::new()` and reuse across all solves; buffer allocation is
/// amortised over the lifetime of the context.
pub struct MetalContext {
    /// The default Metal GPU device (Apple Silicon M-series).
    pub device:         Device,
    /// Command queue used to encode and submit compute work.
    pub queue:          CommandQueue,
    // ── single-airfoil path (original) ───────────────────────────────────────
    /// Compute pipeline for `panel_influence_2d` (single-airfoil kernel).
    pub panel_pipeline: ComputePipelineState,
    /// Shared geometry buffer: `PanelNode[IQX]` — written by CPU, read by GPU.
    pub node_buf:       Buffer,
    /// Shared AIJ buffer: `f32[IQX*(IQX+1)]` — written by GPU, read back by CPU.
    pub aij_buf:        Buffer,
    // ── batch path ───────────────────────────────────────────────────────────
    /// Compute pipeline for `panel_influence_batch` (batched multi-airfoil kernel).
    pub batch_pipeline: ComputePipelineState,
    /// Batched geometry buffer: `PanelNode[MAX_BATCH * NODE_STRIDE]`.
    pub batch_node_buf: Buffer,
    /// Batched AIJ buffer: `f32[MAX_BATCH * AIJ_STRIDE]`.
    pub batch_aij_buf:  Buffer,
    /// Per-airfoil TE cosine factor `scs`: `f32[MAX_BATCH]`.
    pub batch_scs_buf:  Buffer,
    /// Per-airfoil TE sine factor `sds`: `f32[MAX_BATCH]`.
    pub batch_sds_buf:  Buffer,
}

impl MetalContext {
    /// Initialise Metal device, command queue, compute pipelines, and pre-allocated shared buffers.
    /// Panics if no Metal-capable GPU is found or the compiled `.metallib` cannot be loaded.
    pub fn new() -> Self {
        let device = Device::system_default()
            .expect("no Metal-capable GPU found");
        let queue  = device.new_command_queue();

        let shader_dir = env!("METAL_SHADER_DIR");
        let lib = load_metallib(&device,
            &format!("{shader_dir}/panel_influence.metallib"));

        let panel_pipeline = make_pipeline(&device, &lib, "panel_influence_2d");
        let batch_pipeline = make_pipeline(&device, &lib, "panel_influence_batch");

        let opts  = MTLResourceOptions::StorageModeShared;
        let nn    = IQX + 1;

        // Single buffers
        let node_buf = device.new_buffer(
            (IQX * std::mem::size_of::<PanelNode>()) as u64, opts);
        let aij_buf  = device.new_buffer(
            (nn * nn * std::mem::size_of::<f32>()) as u64, opts);

        // Batch buffers
        let batch_node_buf = device.new_buffer(
            (MAX_BATCH * NODE_STRIDE * std::mem::size_of::<PanelNode>()) as u64, opts);
        let batch_aij_buf  = device.new_buffer(
            (MAX_BATCH * AIJ_STRIDE  * std::mem::size_of::<f32>()) as u64, opts);
        let batch_scs_buf  = device.new_buffer(
            (MAX_BATCH * std::mem::size_of::<f32>()) as u64, opts);
        let batch_sds_buf  = device.new_buffer(
            (MAX_BATCH * std::mem::size_of::<f32>()) as u64, opts);

        Self {
            device, queue,
            panel_pipeline, node_buf, aij_buf,
            batch_pipeline, batch_node_buf, batch_aij_buf,
            batch_scs_buf, batch_sds_buf,
        }
    }

    pub fn upload_nodes(&self, nodes: &[PanelNode]) {
        let ptr = self.node_buf.contents() as *mut PanelNode;
        unsafe { std::ptr::copy_nonoverlapping(nodes.as_ptr(), ptr, nodes.len()); }
    }

    /// Readback from the single-airfoil aij_buf.
    pub fn readback_aij(&self, aij_out: &mut [f64], n: usize, nn: usize) {
        self.readback_aij_from(self.aij_buf.contents() as *const f32, 0, aij_out, n, nn);
    }

    /// Readback airfoil `ai` from the batch aij_buf.
    pub fn readback_batch_aij(&self, ai: usize, aij_out: &mut [f64], n: usize, nn: usize) {
        let base = self.batch_aij_buf.contents() as *const f32;
        self.readback_aij_from(base, ai * AIJ_STRIDE, aij_out, n, nn);
    }

    fn readback_aij_from(&self, ptr: *const f32, offset: usize,
                          aij_out: &mut [f64], n: usize, nn: usize) {
        for i in 0..n {
            for j in 0..nn {
                aij_out[i * nn + j] =
                    unsafe { *ptr.add(offset + i * nn + j) } as f64;
            }
        }
    }
}

fn load_metallib(device: &Device, path: &str) -> Library {
    device.new_library_with_file(path)
        .unwrap_or_else(|e| panic!("failed to load metallib {path}: {e}"))
}

fn make_pipeline(device: &Device, lib: &Library, name: &str) -> ComputePipelineState {
    let func = lib.get_function(name, None)
        .unwrap_or_else(|e| panic!("kernel '{name}' not found: {e}"));
    device.new_compute_pipeline_state_with_function(&func)
        .unwrap_or_else(|e| panic!("pipeline for '{name}' failed: {e}"))
}
