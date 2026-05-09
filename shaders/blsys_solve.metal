#include <metal_stdlib>
using namespace metal;

// Solve one 4×4 system (packed as 4×5: coefficients + RHS) in registers.
// One thread per BL station — ~279 independent solves per Newton step.
kernel void blsys_solve(
    device const float* systems [[buffer(0)]],  // [nbl × 20] packed 4×5
    device       float* deltas  [[buffer(1)]],  // [nbl × 4]  output corrections
    uint gid [[thread_position_in_grid]])
{
    // Load 4×5 system into registers
    float a[4][5];
    uint base = gid * 20;
    for (uint r = 0; r < 4; r++)
        for (uint c = 0; c < 5; c++)
            a[r][c] = systems[base + r * 5 + c];

    // TODO: 4×4 Gaussian elimination with partial pivoting (fully in registers)
    // Write 4 corrections to deltas[gid*4 .. gid*4+3]
    for (uint r = 0; r < 4; r++)
        deltas[gid * 4 + r] = 0.0f;

    (void)a;
}
