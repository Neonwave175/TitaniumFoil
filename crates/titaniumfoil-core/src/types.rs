// Data structures derived from XFOIL.INC and XBL.INC COMMON blocks.
// All Fortran COMMON blocks are replaced by fields on XfoilState,
// passed as &mut XfoilState through every module.

// ── Array dimensions (from XFOIL.INC PARAMETER statements) ──────────────────
pub const IQX: usize = 360; // panel nodes on airfoil
pub const IWX: usize = 49;  // wake panel nodes
pub const IZX: usize = 409; // IQX + IWX
pub const IVX: usize = 279; // BL nodes per side  (IQX/2 + IWX + 50)
pub const ISX: usize = 2;   // surfaces: 0=upper, 1=lower
pub const IBX: usize = 1440; // 4*IQX
pub const NAX: usize = 800; // max polar accumulation points
pub const NPX: usize = 12;  // max stored polars

// ── COMMON/CR05/ — airfoil + wake geometry ──────────────────────────────────
#[derive(Debug, Clone)]
pub struct PanelGeometry {
    pub x:      Vec<f64>, // [IZX] panel node x-coords
    pub y:      Vec<f64>, // [IZX] panel node y-coords
    pub xp:     Vec<f64>, // [IZX] dx/ds
    pub yp:     Vec<f64>, // [IZX] dy/ds
    pub s:      Vec<f64>, // [IZX] arc length
    pub nx:     Vec<f64>, // [IZX] outward normal x
    pub ny:     Vec<f64>, // [IZX] outward normal y
    pub apanel: Vec<f64>, // [IZX] panel angle
    pub wgap:   Vec<f64>, // [IWX] wake gap (TE thickness)
    pub sle:    f64,      // arc length at leading edge
    pub xle:    f64,
    pub yle:    f64,
    pub xte:    f64,
    pub yte:    f64,
    pub chord:  f64,
    pub waklen: f64,
    pub n:      usize,    // number of airfoil panels
    pub nw:     usize,    // number of wake panels
}

impl Default for PanelGeometry {
    fn default() -> Self {
        Self {
            x:      vec![0.0; IZX],
            y:      vec![0.0; IZX],
            xp:     vec![0.0; IZX],
            yp:     vec![0.0; IZX],
            s:      vec![0.0; IZX],
            nx:     vec![0.0; IZX],
            ny:     vec![0.0; IZX],
            apanel: vec![0.0; IZX],
            wgap:   vec![0.0; IWX],
            sle: 0.0, xle: 0.0, yle: 0.0,
            xte: 1.0, yte: 0.0, chord: 1.0,
            waklen: 1.0,
            n: 0, nw: 0,
        }
    }
}

// ── COMMON/CR03/ + COMMON/CR06/ — inviscid panel solution ───────────────────
#[derive(Debug, Clone)]
pub struct PanelState {
    pub aij:          Vec<f64>, // [IQX×IQX] vortex influence matrix (LU factored in place)
    pub dij:          Vec<f64>, // [IZX×IZX] source influence matrix
    pub aij_piv:      Vec<i32>, // [IQX] LU pivot indices
    pub gam:          Vec<f64>, // [IQX] vortex panel strengths at current α
    pub gamu:         Vec<f64>, // [IQX×2] basis solutions α=0° and α=90°
    pub gam_a:        Vec<f64>, // [IQX] dγ/dα
    pub sig:          Vec<f64>, // [IZX] source distribution
    pub aij_factored: bool,
    pub dij_built:    bool,
}

impl Default for PanelState {
    fn default() -> Self {
        Self {
            aij:          vec![0.0; IQX * IQX],
            dij:          vec![0.0; IZX * IZX],
            aij_piv:      vec![0; IQX],
            gam:          vec![0.0; IQX],
            gamu:         vec![0.0; IQX * 2],
            gam_a:        vec![0.0; IQX],
            sig:          vec![0.0; IZX],
            aij_factored: false,
            dij_built:    false,
        }
    }
}

// ── COMMON/CR04/ — surface velocities and pressure ──────────────────────────
#[derive(Debug, Clone)]
pub struct VelocityField {
    pub qinv:   Vec<f64>, // [IZX] inviscid surface speed
    pub qvis:   Vec<f64>, // [IZX] viscous surface speed
    pub cpi:    Vec<f64>, // [IZX] inviscid Cp
    pub cpv:    Vec<f64>, // [IZX] viscous Cp
    pub qinvu:  Vec<f64>, // [IZX×2] basis speeds for α=0°,90°
    pub qinv_a: Vec<f64>, // [IZX] dQ/dα
}

impl Default for VelocityField {
    fn default() -> Self {
        Self {
            qinv:   vec![0.0; IZX],
            qvis:   vec![0.0; IZX],
            cpi:    vec![0.0; IZX],
            cpv:    vec![0.0; IZX],
            qinvu:  vec![0.0; IZX * 2],
            qinv_a: vec![0.0; IZX],
        }
    }
}

// ── XBL.INC /V_VAR1/ + /V_VAR2/ — one BL station's full state ───────────────
#[derive(Debug, Clone, Default)]
pub struct BLStation {
    // primary integration variables
    pub x:  f64, // arc length coordinate
    pub u:  f64, // edge velocity
    pub t:  f64, // momentum thickness θ
    pub d:  f64, // displacement thickness δ*
    pub s:  f64, // kinetic energy thickness θ*
    pub h:  f64, // shape parameter H = δ*/θ
    pub m:  f64, // mass defect m = ρ·Ue·δ*
    pub r:  f64, // density ratio ρ/ρ∞

    // secondary variables (computed by BLVAR)
    pub hk: f64, // kinematic shape parameter
    pub hs: f64, // energy shape parameter
    pub hc: f64, // density shape parameter
    pub rt: f64, // momentum-thickness Reynolds number
    pub cf: f64, // skin friction coefficient
    pub di: f64, // dissipation coefficient
    pub us: f64, // slip velocity
    pub cq: f64, // shear stress lag coefficient
    pub de: f64, // max equilibrium δ*

    // Newton system sensitivities (∂secondary/∂primary)
    pub cf_u: f64, pub cf_t: f64, pub cf_d: f64, pub cf_m: f64, pub cf_r: f64,
    pub di_u: f64, pub di_t: f64, pub di_d: f64, pub di_m: f64, pub di_r: f64,
    pub hk_u: f64, pub hk_t: f64, pub hk_d: f64, pub hk_m: f64, pub hk_r: f64,
    pub hs_u: f64, pub hs_t: f64, pub hs_d: f64, pub hs_m: f64, pub hs_r: f64,
    pub rt_u: f64, pub rt_t: f64,
}

// ── COMMON/CR15/ — per-surface BL arrays ────────────────────────────────────
#[derive(Debug, Clone)]
pub struct BLSurface {
    pub sta:  Vec<BLStation>, // [IVX]
    pub xssi: Vec<f64>,       // [IVX] arc-length coordinate along surface
    pub uedg: Vec<f64>,       // [IVX] boundary-layer edge velocity
    pub uinv: Vec<f64>,       // [IVX] inviscid edge velocity
    pub mass: Vec<f64>,       // [IVX] mass defect
    pub thet: Vec<f64>,       // [IVX] momentum thickness
    pub dstr: Vec<f64>,       // [IVX] displacement thickness
    pub ctau: Vec<f64>,       // [IVX] shear stress parameter √(Cτ)
    pub ipan: Vec<usize>,     // [IVX] panel index for each BL station
    pub nbl:   usize,         // number of active BL stations
    pub iblte: usize,         // BL index of trailing edge
    pub itran: usize,         // BL index of transition point
}

impl Default for BLSurface {
    fn default() -> Self {
        Self {
            sta:  vec![BLStation::default(); IVX],
            xssi: vec![0.0; IVX],
            uedg: vec![0.0; IVX],
            uinv: vec![0.0; IVX],
            mass: vec![0.0; IVX],
            thet: vec![0.0; IVX],
            dstr: vec![0.0; IVX],
            ctau: vec![0.0; IVX],
            ipan: vec![0; IVX],
            nbl: 0, iblte: 0, itran: 0,
        }
    }
}

// ── XBL.INC /V_SYS/ — Newton system matrices (per BLSYS call) ───────────────
#[derive(Debug, Clone, Default)]
pub struct BLSystem {
    pub vs1:   [[f64; 5]; 4], // 4×5 left station coefficients
    pub vs2:   [[f64; 5]; 4], // 4×5 right station coefficients
    pub vsrez: [f64; 4],      // residuals
    pub vsr:   [f64; 4],
    pub vsm:   [f64; 4],
    pub vsx:   [f64; 4],
}

// ── BLPAR.INC — BL closure model constants ───────────────────────────────────
#[derive(Debug, Clone)]
pub struct BLParams {
    pub sccon:  f64, // laminar stability constant
    pub gacon:  f64,
    pub gbcon:  f64,
    pub gccon:  f64,
    pub dlcon:  f64,
    pub ctcon:  f64, // turbulence model constant
    pub ctrcex: f64,
    pub duxcon: f64,
    pub cffac:  f64,
    // compressibility
    pub tklam:     f64,
    pub tkl_msq:   f64,
    pub gambl:     f64,
    pub gm1bl:     f64,
    pub tkbl:      f64,
    pub tkbl_ms:   f64,
    pub rstbl:     f64,
    pub rstbl_ms:  f64,
    pub hstinv:    f64,
    pub hstinv_ms: f64,
    pub reybl:     f64,
    pub reybl_re:  f64,
    pub reybl_ms:  f64,
    pub amcrit:    f64,
    pub acrit:     f64, // amplification factor for transition (default 9)
    pub idampv:    i32, // damping model selector
}

impl Default for BLParams {
    fn default() -> Self {
        Self {
            sccon: 5.6, gacon: 6.70, gbcon: 0.75, gccon: 18.0,
            dlcon: 0.9, ctcon: 0.03, ctrcex: 1.8, duxcon: 0.1, cffac: 1.0,
            tklam: 0.0, tkl_msq: 0.0, gambl: 1.4, gm1bl: 0.4,
            tkbl: 0.0, tkbl_ms: 0.0, rstbl: 0.0, rstbl_ms: 0.0,
            hstinv: 0.0, hstinv_ms: 0.0,
            reybl: 0.0, reybl_re: 0.0, reybl_ms: 0.0,
            amcrit: 0.0, acrit: 9.0, idampv: 1,
        }
    }
}

// ── COMMON/CR09/ — operating point ───────────────────────────────────────────
#[derive(Debug, Clone, Default)]
pub struct OperatingPoint {
    pub alfa:  f64, // angle of attack (radians)
    pub adeg:  f64, // angle of attack (degrees)
    pub cl:    f64,
    pub cm:    f64,
    pub cd:    f64,
    pub cdp:   f64, // pressure drag
    pub cdf:   f64, // friction drag
    pub minf:  f64, // freestream Mach number
    pub reinf: f64, // Reynolds number
    pub qinf:  f64, // reference speed (normalised = 1.0)
    pub circ:  f64, // circulation
    pub converged: bool,
}

// ── COMMON/VMAT/ — block-tridiagonal BL Newton system matrices ───────────────
// Needed by linalg::blsolv. Indexed as [row][col][station] matching Fortran
// VA(3,2,IZX), VB(3,2,IZX), VDEL(3,2,IZX), VM(3,IZX,IZX).
// We use a helper Index3 type to keep access readable.
#[derive(Debug, Clone)]
pub struct Idx3<const R: usize, const C: usize, const D: usize>(pub Vec<f64>);

impl<const R: usize, const C: usize, const D: usize> Idx3<R, C, D> {
    pub fn new() -> Self { Self(vec![0.0; R * C * D]) }
}

impl<const R: usize, const C: usize, const D: usize> std::ops::Index<[usize; 3]>
    for Idx3<R, C, D>
{
    type Output = f64;
    fn index(&self, [r, c, d]: [usize; 3]) -> &f64 { &self.0[r * C * D + c * D + d] }
}

impl<const R: usize, const C: usize, const D: usize> std::ops::IndexMut<[usize; 3]>
    for Idx3<R, C, D>
{
    fn index_mut(&mut self, [r, c, d]: [usize; 3]) -> &mut f64 {
        &mut self.0[r * C * D + c * D + d]
    }
}

#[derive(Debug, Clone)]
pub struct BLMatrices {
    pub va:    Idx3<3, 2, IZX>,      // 3×2×IZX diagonal blocks
    pub vb:    Idx3<3, 2, IZX>,      // 3×2×IZX sub-diagonal blocks
    pub vz:    [[f64; 2]; 3],        // 3×2 TE coupling block
    pub vdel:  Idx3<3, 2, IZX>,      // 3×2×IZX RHS/solution
    pub vm:    Vec<f64>,              // 3×IZX×IZX mass influence (flat)
    pub isys:  Vec<[usize; 2]>,      // IVX×ISX station→system index map
    pub iblte: [usize; 2],           // BL TE station index per surface
    pub nsys:  usize,                // total number of coupled stations
    pub vaccel: f64,                 // threshold for VM sparsification
    pub s_span: f64,                 // S(N)-S(1) arc length span
}

impl BLMatrices {
    /// Access/mutate VM[row, col_station, row_station] — VM(3,IZX,IZX) in Fortran.
    #[inline] pub fn vm(&self, r: usize, l: usize, iv: usize) -> f64 {
        self.vm[r * IZX * IZX + l * IZX + iv]
    }
    #[inline] pub fn vm_mut(&mut self, r: usize, l: usize, iv: usize) -> &mut f64 {
        &mut self.vm[r * IZX * IZX + l * IZX + iv]
    }
}

impl Default for BLMatrices {
    fn default() -> Self {
        Self {
            va:     Idx3::new(),
            vb:     Idx3::new(),
            vz:     [[0.0; 2]; 3],
            vdel:   Idx3::new(),
            vm:     vec![0.0; 3 * IZX * IZX],
            isys:   vec![[0; 2]; IVX],
            iblte:  [0; 2],
            nsys:   0,
            vaccel: 1e-8,
            s_span: 1.0,
        }
    }
}

// ── FlowState: flags, work arrays and coupling fields ────────────────────────
// Covers COMMON/QMAT/, COMMON/CR07/ flags, BL coupling arrays, etc.
#[derive(Debug, Clone)]
pub struct FlowState {
    // flags
    pub sharp:   bool,   // sharp trailing edge
    pub limage:  bool,   // ground image flag
    pub lvisc:   bool,   // viscous mode active
    pub lalfa:   bool,   // alpha prescribed (vs CL)
    pub lblini:  bool,   // BL initialized
    pub lqaij:   bool,   // AIJ factored
    pub ladij:   bool,   // DIJ built

    // TE geometry (COMMON/CR06/)
    pub ante:   f64,   // TE base height (anti-symmetric)
    pub aste:   f64,   // TE base height (symmetric)
    pub dste:   f64,   // TE thickness
    pub gamte:  f64,   // TE vortex strength
    pub sigte:  f64,   // TE source strength
    pub psio:   f64,   // stagnation streamfunction

    // sensitivity vectors (COMMON/QMAT/) — length IQX or IZX
    pub dzdg:   Vec<f64>,   // [IQX] dPsi/dGamma
    pub dzdn:   Vec<f64>,   // [IQX] dPsi/dn
    pub dzdm:   Vec<f64>,   // [IZX] dPsi/dSigma
    pub dqdg:   Vec<f64>,   // [IQX] dQ/dGamma
    pub dqdm:   Vec<f64>,   // [IZX] dQ/dSigma
    pub z_qinf: f64,
    pub z_alfa: f64,
    pub z_qdof0: f64,
    pub z_qdof1: f64,

    // BIJ work array [IQX × IZX] for GGCALC / QDCALC
    pub bij:    Vec<f64>,

    // viscous-inviscid coupling
    pub ipan:   Vec<[usize; ISX]>,  // [IVX] panel index per BL station/surface
    pub isys:   Vec<[usize; ISX]>,  // [IVX] Newton-system index
    pub iblte:  [usize; ISX],
    pub itran:  [usize; ISX],
    pub nbl:    [usize; ISX],

    // VTI: tangent velocity direction at BL station (IVX × ISX)
    pub vti:    Vec<f64>,           // [IVX × ISX]
    pub uinv_a: Vec<f64>,           // [IVX × ISX] dUinv/dAlfa

    // NSYS: total coupled Newton system size
    pub nsys:   usize,

    // transition info
    pub xiforc: [f64; ISX],
    pub amcrit: f64,
    pub clspec: f64,
}

impl Default for FlowState {
    fn default() -> Self {
        Self {
            sharp:  false, limage: false, lvisc: false,
            lalfa:  true,  lblini: false, lqaij: false, ladij: false,
            ante: 0.0, aste: 0.0, dste: 0.0, gamte: 0.0, sigte: 0.0, psio: 0.0,
            dzdg:   vec![0.0; IQX],
            dzdn:   vec![0.0; IQX],
            dzdm:   vec![0.0; IZX],
            dqdg:   vec![0.0; IQX],
            dqdm:   vec![0.0; IZX],
            z_qinf: 0.0, z_alfa: 0.0, z_qdof0: 0.0, z_qdof1: 0.0,
            bij:    vec![0.0; IQX * IZX],
            ipan:   vec![[0; ISX]; IVX],
            isys:   vec![[0; ISX]; IVX],
            iblte:  [0; ISX],
            itran:  [0; ISX],
            nbl:    [0; ISX],
            vti:    vec![0.0; IVX * ISX],
            uinv_a: vec![0.0; IVX * ISX],
            nsys:   0,
            xiforc: [2.0; ISX],   // default: no forced transition
            amcrit: 0.0,
            clspec: 0.0,
        }
    }
}

// ── Top-level state: replaces all Fortran COMMON globals ─────────────────────
#[derive(Debug, Clone, Default)]
pub struct XfoilState {
    pub geom:   PanelGeometry,
    pub panel:  PanelState,
    pub vel:    VelocityField,
    pub bl:     [BLSurface; ISX],
    pub op:     OperatingPoint,
    pub params: BLParams,
    pub mat:    BLMatrices,
    pub flow:   FlowState,
}
