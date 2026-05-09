// panel_influence.metal
//
// Two kernels sharing the same inner-loop logic via compute_row():
//
//   panel_influence_2d   — single airfoil  (original)
//   panel_influence_batch — B airfoils in one dispatch
//
// Batch dispatch: grid=(B, N, 1)  tg=(T,1,1)
//   tgid.x = airfoil index
//   tgid.y = evaluation row i
//
// Fixed strides (IQX=360):
//   node stride per airfoil = IQX
//   aij  stride per airfoil = IQX * (IQX+1) = 129960

#include <metal_stdlib>
using namespace metal;

constant float QOPI = 0.25f / 3.14159265358979f;
constant float HOPI = 0.50f / 3.14159265358979f;

// Fixed stride constants (IQX = 360)
constant uint kNodeStride = 360u;
constant uint kAijStride  = 360u * 361u;   // 129 960

struct PanelNode {
    float2 xy;
    float2 nxy;
    float  apanel;
    float  s;
    float  _pad[2];
};

// ── Shared inner loop ─────────────────────────────────────────────────────────
// Called from both kernels. Writes one row i of aij[] for the given airfoil.

static inline void compute_row(
    uint i, uint N, uint nn,
    float scs, float sds,
    device const PanelNode* nodes,
    device       float*     aij,
    threadgroup  float*     s_jo,
    threadgroup  float*     s_jp,
    uint tid)
{
    float xi = nodes[i].xy.x;
    float yi = nodes[i].xy.y;
    float cjo = 0.0f, cjp = 0.0f;

    if (tid < N) {
        uint jo = tid;
        uint jp = (jo == N - 1u) ? 0u : jo + 1u;

        float2 dxy = nodes[jp].xy - nodes[jo].xy;
        float  dso = length(dxy);

        if (dso > 0.0f && jo < N - 1u) {
            float dsio = 1.0f / dso;
            float sx = dxy.x * dsio, sy = dxy.y * dsio;
            float rx1 = xi - nodes[jo].xy.x, ry1 = yi - nodes[jo].xy.y;
            float rx2 = xi - nodes[jp].xy.x, ry2 = yi - nodes[jp].xy.y;
            float x1 = sx*rx1 + sy*ry1, x2 = sx*rx2 + sy*ry2;
            float yy = sx*ry1 - sy*rx1;
            float rs1 = rx1*rx1 + ry1*ry1, rs2 = rx2*rx2 + ry2*ry2;
            float g1 = (i != jo && rs1 > 0.0f) ? log(rs1) : 0.0f;
            float t1 = (i != jo && rs1 > 0.0f) ? atan2(x1, yy) : 0.0f;
            float g2 = (i != jp && rs2 > 0.0f) ? log(rs2) : 0.0f;
            float t2 = (i != jp && rs2 > 0.0f) ? atan2(x2, yy) : 0.0f;
            float dxinv = 1.0f / (x1 - x2);
            float psis  = 0.5f*x1*g1 - 0.5f*x2*g2 + x2 - x1 + yy*(t1-t2);
            float psid  = ((x1+x2)*psis + 0.5f*(rs2*g2 - rs1*g1
                           + x1*x1 - x2*x2)) * dxinv;
            cjo = QOPI * (psis - psid);
            cjp = QOPI * (psis + psid);

        } else if (jo == N - 1u && dso > 0.0f) {
            float dsio = 1.0f / dso;
            float sx = dxy.x * dsio, sy = dxy.y * dsio;
            float rx1 = xi - nodes[jo].xy.x, ry1 = yi - nodes[jo].xy.y;
            float rx2 = xi - nodes[jp].xy.x, ry2 = yi - nodes[jp].xy.y;
            float x1 = sx*rx1 + sy*ry1, x2 = sx*rx2 + sy*ry2;
            float yy = sx*ry1 - sy*rx1;
            float rs1 = rx1*rx1 + ry1*ry1, rs2 = rx2*rx2 + ry2*ry2;
            float apan = nodes[jo].apanel;
            float g1 = (i != jo && rs1 > 0.0f) ? log(rs1) : 0.0f;
            float t1 = (i != jo && rs1 > 0.0f) ? atan2(x1, yy) : 0.0f;
            float g2 = (i != jp && rs2 > 0.0f) ? log(rs2) : 0.0f;
            float t2 = (i != jp && rs2 > 0.0f) ? atan2(x2, yy) : 0.0f;
            float psig = 0.5f*yy*(g1-g2) + x2*(t2-apan) - x1*(t1-apan);
            float pgam = 0.5f*x1*g1 - 0.5f*x2*g2 + x2 - x1 + yy*(t1-t2);
            cjo = -HOPI*psig*scs*0.5f + HOPI*pgam*sds*0.5f;
            cjp =  HOPI*psig*scs*0.5f - HOPI*pgam*sds*0.5f;
        }
    }

    if (tid < 384u) { s_jo[tid] = cjo; s_jp[tid] = cjp; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < N) {
        uint prev = (tid == 0u) ? (N - 1u) : (tid - 1u);
        aij[i * nn + tid] = s_jo[tid] + s_jp[prev];
    }
    if (tid == 0u) aij[i * nn + N] = -1.0f;
}

// ── Single-airfoil kernel (unchanged API) ─────────────────────────────────────
kernel void panel_influence_2d(
    device const PanelNode* nodes [[buffer(0)]],
    device       float*     aij   [[buffer(1)]],
    constant     uint&      N     [[buffer(2)]],
    constant     uint&      nn    [[buffer(3)]],
    constant     float&     scs   [[buffer(4)]],
    constant     float&     sds   [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    threadgroup float s_jo[384];
    threadgroup float s_jp[384];
    uint i = tgid.y;
    if (i >= N) return;
    compute_row(i, N, nn, scs, sds, nodes, aij, s_jo, s_jp, tid);
}

// ── Batch kernel: B airfoils in one dispatch ──────────────────────────────────
//
// buffer(0): PanelNode[B * kNodeStride]   — all geometries
// buffer(1): float[B * kAijStride]        — all output matrices
// buffer(2): float[B]                     — scs per airfoil
// buffer(3): float[B]                     — sds per airfoil
// buffer(4): N  (same for every airfoil in this batch)
// buffer(5): nn = N+1
//
// grid  = (B, N, 1)   tg = (T, 1, 1)
kernel void panel_influence_batch(
    device const PanelNode* all_nodes [[buffer(0)]],
    device       float*     all_aij   [[buffer(1)]],
    device const float*     scs_arr   [[buffer(2)]],
    device const float*     sds_arr   [[buffer(3)]],
    constant     uint&      N         [[buffer(4)]],
    constant     uint&      nn        [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    threadgroup float s_jo[384];
    threadgroup float s_jp[384];

    uint airfoil = tgid.x;
    uint i       = tgid.y;
    if (i >= N) return;

    device const PanelNode* nodes = all_nodes + airfoil * kNodeStride;
    device       float*     aij   = all_aij   + airfoil * kAijStride;
    float scs = scs_arr[airfoil];
    float sds = sds_arr[airfoil];

    compute_row(i, N, nn, scs, sds, nodes, aij, s_jo, s_jp, tid);
}
