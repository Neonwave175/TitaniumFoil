// GPU dispatch for per-station BL variable evaluation and 4×4 solves.

use titaniumfoil_core::types::XfoilState;
use crate::context::MetalContext;

pub fn compute_blvar(_ctx: &MetalContext, _state: &mut XfoilState, _is: usize) {
    // TODO: upload BL station arrays to MTLBuffer
    // TODO: dispatch blvar_compute kernel (1D, threads = nbl)
    // TODO: read back secondary variables into state.bl[is].sta
}

pub fn solve_blsys(_ctx: &MetalContext, _state: &mut XfoilState, _is: usize) {
    // TODO: upload packed 4×5 systems to MTLBuffer
    // TODO: dispatch blsys_solve kernel (1D, threads = nbl)
    // TODO: read back correction vectors
}
