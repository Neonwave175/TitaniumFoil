#include <metal_stdlib>
using namespace metal;

struct BLStationIn {
    float uedg; // edge velocity
    float mass; // mass defect
    float thet; // momentum thickness
    float dstr; // displacement thickness
    float ctau; // shear stress parameter
};

struct BLStationOut {
    float hk, hs, hc;
    float rt, cf, di;
    float us, cq, de;
    // sensitivities
    float cf_u, cf_t, cf_d;
    float di_u, di_t, di_d;
    float hk_u, hk_t, hk_d;
};

struct BLParamsGPU {
    float gacon, gbcon, gccon;
    float ctcon, dlcon, sccon;
    float tklam, tkbl, rstbl, hstinv, reybl;
    float gambl, amcrit, acrit;
};

// One thread per BL station — all stations independent within one Newton step.
kernel void blvar_compute(
    device const BLStationIn*  inp    [[buffer(0)]],
    device       BLStationOut* out    [[buffer(1)]],
    constant     BLParamsGPU&  params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    BLStationIn  s  = inp[gid];
    BLStationOut r;

    // TODO: implement BLVAR secondary variable calculations
    r.hk = 0.0f; r.hs = 0.0f; r.hc = 0.0f;
    r.rt = 0.0f; r.cf = 0.0f; r.di = 0.0f;
    r.us = 0.0f; r.cq = 0.0f; r.de = 0.0f;
    r.cf_u = 0.0f; r.cf_t = 0.0f; r.cf_d = 0.0f;
    r.di_u = 0.0f; r.di_t = 0.0f; r.di_d = 0.0f;
    r.hk_u = 0.0f; r.hk_t = 0.0f; r.hk_d = 0.0f;

    out[gid] = r;
    (void)s; (void)params;
}
