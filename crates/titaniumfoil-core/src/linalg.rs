// xsolve.f → Rust (direct port, Mark Drela MIT XFOIL)
// All matrices are row-major (Rust default). Fortran used column-major,
// but the algorithms are index-symmetric so the port is direct.

use crate::types::BLMatrices;

// ── General dense solvers ─────────────────────────────────────────────────────

/// Gaussian elimination with partial pivoting.
/// `z`: N×N coefficient matrix (row-major, stride=nn), destroyed.
/// `r`: N×NRHS right-hand sides, overwritten with solutions.
pub fn gauss(nn: usize, z: &mut [f64], r: &mut [f64], nrhs: usize) {
    for np in 0..nn - 1 {
        let mut nx = np;
        for n in np + 1..nn {
            if z[n * nn + np].abs() > z[nx * nn + np].abs() { nx = n; }
        }

        let pivot = 1.0 / z[nx * nn + np];
        z[nx * nn + np] = z[np * nn + np];

        for l in np + 1..nn {
            let tmp = z[nx * nn + l] * pivot;
            z[nx * nn + l] = z[np * nn + l];
            z[np * nn + l] = tmp;
        }
        for l in 0..nrhs {
            let tmp = r[nx * nrhs + l] * pivot;
            r[nx * nrhs + l] = r[np * nrhs + l];
            r[np * nrhs + l] = tmp;
        }

        for k in np + 1..nn {
            let ztmp = z[k * nn + np];
            for l in np + 1..nn { z[k * nn + l] -= ztmp * z[np * nn + l]; }
            for l in 0..nrhs   { r[k * nrhs + l] -= ztmp * r[np * nrhs + l]; }
        }
    }

    for l in 0..nrhs {
        r[(nn - 1) * nrhs + l] /= z[(nn - 1) * nn + (nn - 1)];
    }

    for np in (0..nn - 1).rev() {
        for l in 0..nrhs {
            for k in np + 1..nn {
                r[np * nrhs + l] -= z[np * nn + k] * r[k * nrhs + l];
            }
        }
    }
}

/// Solve a single 4×4 system with 1 RHS, entirely in registers.
/// `sys[row]` = [a0, a1, a2, a3, rhs]. Returns solution [x0..x3].
pub fn gauss4(sys: &mut [[f64; 5]; 4]) -> [f64; 4] {
    const N: usize = 4;
    for np in 0..N - 1 {
        let mut nx = np;
        for n in np + 1..N {
            if sys[n][np].abs() > sys[nx][np].abs() { nx = n; }
        }
        let pivot = 1.0 / sys[nx][np];
        sys[nx][np] = sys[np][np];
        for l in np + 1..=N {
            let tmp = sys[nx][l] * pivot;
            sys[nx][l] = sys[np][l];
            sys[np][l] = tmp;
        }
        for k in np + 1..N {
            let ztmp = sys[k][np];
            for l in np + 1..=N { sys[k][l] -= ztmp * sys[np][l]; }
        }
    }
    sys[N - 1][N] /= sys[N - 1][N - 1];
    for np in (0..N - 1).rev() {
        for k in np + 1..N { sys[np][N] -= sys[np][k] * sys[k][N]; }
    }
    [sys[0][N], sys[1][N], sys[2][N], sys[3][N]]
}

// ── LU decomposition ─────────────────────────────────────────────────────────

/// LU decomposition with implicit partial pivoting (Crout).
/// `a`: N×N row-major, stride=`stride`. Replaced with LU factors.
/// `piv`: output pivot indices (length N).
pub fn ludcmp(a: &mut [f64], n: usize, stride: usize, piv: &mut [i32]) {
    let mut vv = vec![0.0f64; n];
    for i in 0..n {
        let aamax = (0..n).map(|j| a[i * stride + j].abs()).fold(0.0f64, f64::max);
        vv[i] = 1.0 / aamax;
    }
    for j in 0..n {
        for i in 0..j {
            let mut sum = a[i * stride + j];
            for k in 0..i { sum -= a[i * stride + k] * a[k * stride + j]; }
            a[i * stride + j] = sum;
        }
        let mut aamax = 0.0f64;
        let mut imax  = j;
        for i in j..n {
            let mut sum = a[i * stride + j];
            for k in 0..j { sum -= a[i * stride + k] * a[k * stride + j]; }
            a[i * stride + j] = sum;
            let dum = vv[i] * sum.abs();
            if dum >= aamax { imax = i; aamax = dum; }
        }
        if j != imax {
            for k in 0..n { a.swap(imax * stride + k, j * stride + k); }
            vv[imax] = vv[j];
        }
        piv[j] = imax as i32;
        if j != n - 1 {
            let dum = 1.0 / a[j * stride + j];
            for i in j + 1..n { a[i * stride + j] *= dum; }
        }
    }
}

/// Back-substitution after ludcmp. `b` is RHS → solution.
pub fn baksub(a: &[f64], n: usize, stride: usize, piv: &[i32], b: &mut [f64]) {
    let mut ii: Option<usize> = None;
    for i in 0..n {
        let ll  = piv[i] as usize;
        let mut sum = b[ll];
        b[ll] = b[i];
        if let Some(start) = ii {
            for j in start..i { sum -= a[i * stride + j] * b[j]; }
        } else if sum != 0.0 {
            ii = Some(i);
        }
        b[i] = sum;
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in i + 1..n { sum -= a[i * stride + j] * b[j]; }
        b[i] = sum / a[i * stride + i];
    }
}

// ── Coupled BL block-tridiagonal solver ──────────────────────────────────────

/// Solve the coupled viscous-inviscid Newton system (BLSOLV from xsolve.f).
///
/// Each station has a 3-DOF block: [Ctau, Theta, mass_defect].
/// VA/VB are local 3×2 coefficient blocks; VM is dense 3×NSYS mass coupling.
pub fn blsolv(bl: &mut BLMatrices) {
    let nsys  = bl.nsys;
    let ivte1 = bl.isys[bl.iblte[0]][0];
    let vacc1 = bl.vaccel;
    let vacc2 = bl.vaccel * 2.0 / bl.s_span;
    let vacc3 = bl.vaccel * 2.0 / bl.s_span;

    // ── Phase 1: forward sweep ────────────────────────────────────────────────
    for iv in 0..nsys {
        let ivp = iv + 1;

        // normalise VA row 0 by its diagonal
        let pivot = 1.0 / bl.va[[0, 1, iv]];
        bl.va[[0, 1, iv]] *= pivot;
        for l in iv..nsys { *bl.vm_mut(0, l, iv) *= pivot; }
        bl.vdel[[0, 0, iv]] *= pivot;
        bl.vdel[[0, 1, iv]] *= pivot;

        // eliminate rows 1 and 2 of VA column 0
        for k in 1..3usize {
            let vtmp = bl.va[[k, 0, iv]];
            bl.va[[k, 1, iv]] -= vtmp * bl.va[[0, 1, iv]];
            for l in iv..nsys { *bl.vm_mut(k, l, iv) -= vtmp * bl.vm(0, l, iv); }
            bl.vdel[[k, 0, iv]] -= vtmp * bl.vdel[[0, 0, iv]];
            bl.vdel[[k, 1, iv]] -= vtmp * bl.vdel[[0, 1, iv]];
        }

        // normalise VA row 1
        let pivot = 1.0 / bl.va[[1, 1, iv]];
        for l in iv..nsys { *bl.vm_mut(1, l, iv) *= pivot; }
        bl.vdel[[1, 0, iv]] *= pivot;
        bl.vdel[[1, 1, iv]] *= pivot;

        // eliminate VA row 2 column 1
        let vtmp = bl.va[[2, 1, iv]];
        for l in iv..nsys { *bl.vm_mut(2, l, iv) -= vtmp * bl.vm(1, l, iv); }
        bl.vdel[[2, 0, iv]] -= vtmp * bl.vdel[[1, 0, iv]];
        bl.vdel[[2, 1, iv]] -= vtmp * bl.vdel[[1, 1, iv]];

        // normalise row 2 (mass equation diagonal)
        let pivot = 1.0 / bl.vm(2, iv, iv);
        for l in ivp..nsys { *bl.vm_mut(2, l, iv) *= pivot; }
        bl.vdel[[2, 0, iv]] *= pivot;
        bl.vdel[[2, 1, iv]] *= pivot;

        // back-eliminate rows 0,1 using normalised row 2
        let (v1, v2) = (bl.vm(0, iv, iv), bl.vm(1, iv, iv));
        for l in ivp..nsys {
            *bl.vm_mut(0, l, iv) -= v1 * bl.vm(2, l, iv);
            *bl.vm_mut(1, l, iv) -= v2 * bl.vm(2, l, iv);
        }
        bl.vdel[[0, 0, iv]] -= v1 * bl.vdel[[2, 0, iv]];
        bl.vdel[[1, 0, iv]] -= v2 * bl.vdel[[2, 0, iv]];
        bl.vdel[[0, 1, iv]] -= v1 * bl.vdel[[2, 1, iv]];
        bl.vdel[[1, 1, iv]] -= v2 * bl.vdel[[2, 1, iv]];

        // eliminate VA row 0 column 1
        let vtmp = bl.va[[0, 1, iv]];
        for l in ivp..nsys { *bl.vm_mut(0, l, iv) -= vtmp * bl.vm(1, l, iv); }
        bl.vdel[[0, 0, iv]] -= vtmp * bl.vdel[[1, 0, iv]];
        bl.vdel[[0, 1, iv]] -= vtmp * bl.vdel[[1, 1, iv]];

        if iv == nsys - 1 { continue; }

        // eliminate VB(ivp) into VM(ivp)
        for k in 0..3usize {
            let (v1, v2, v3) = (bl.vb[[k, 0, ivp]], bl.vb[[k, 1, ivp]], bl.vm(k, iv, ivp));
            for l in ivp..nsys {
                *bl.vm_mut(k, l, ivp) -= v1 * bl.vm(0, l, iv)
                                       + v2 * bl.vm(1, l, iv)
                                       + v3 * bl.vm(2, l, iv);
            }
            bl.vdel[[k, 0, ivp]] -= v1 * bl.vdel[[0, 0, iv]]
                                   + v2 * bl.vdel[[1, 0, iv]]
                                   + v3 * bl.vdel[[2, 0, iv]];
            bl.vdel[[k, 1, ivp]] -= v1 * bl.vdel[[0, 1, iv]]
                                   + v2 * bl.vdel[[1, 1, iv]]
                                   + v3 * bl.vdel[[2, 1, iv]];
        }

        // TE coupling: eliminate VZ block into lower surface start
        if iv == ivte1 {
            let ivz = bl.isys[bl.iblte[1] + 1][1];
            for k in 0..3usize {
                let (v1, v2) = (bl.vz[k][0], bl.vz[k][1]);
                for l in ivp..nsys {
                    *bl.vm_mut(k, l, ivz) -= v1 * bl.vm(0, l, iv)
                                           + v2 * bl.vm(1, l, iv);
                }
                bl.vdel[[k, 0, ivz]] -= v1 * bl.vdel[[0, 0, iv]]
                                       + v2 * bl.vdel[[1, 0, iv]];
                bl.vdel[[k, 1, ivz]] -= v1 * bl.vdel[[0, 1, iv]]
                                       + v2 * bl.vdel[[1, 1, iv]];
            }
        }

        if ivp == nsys - 1 { continue; }

        // sparsified elimination of lower VM columns (mass coupling)
        for kv in iv + 2..nsys {
            let (v1, v2, v3) = (bl.vm(0, iv, kv), bl.vm(1, iv, kv), bl.vm(2, iv, kv));
            if v1.abs() > vacc1 {
                for l in ivp..nsys { *bl.vm_mut(0, l, kv) -= v1 * bl.vm(2, l, iv); }
                bl.vdel[[0, 0, kv]] -= v1 * bl.vdel[[2, 0, iv]];
                bl.vdel[[0, 1, kv]] -= v1 * bl.vdel[[2, 1, iv]];
            }
            if v2.abs() > vacc2 {
                for l in ivp..nsys { *bl.vm_mut(1, l, kv) -= v2 * bl.vm(2, l, iv); }
                bl.vdel[[1, 0, kv]] -= v2 * bl.vdel[[2, 0, iv]];
                bl.vdel[[1, 1, kv]] -= v2 * bl.vdel[[2, 1, iv]];
            }
            if v3.abs() > vacc3 {
                for l in ivp..nsys { *bl.vm_mut(2, l, kv) -= v3 * bl.vm(2, l, iv); }
                bl.vdel[[2, 0, kv]] -= v3 * bl.vdel[[2, 0, iv]];
                bl.vdel[[2, 1, kv]] -= v3 * bl.vdel[[2, 1, iv]];
            }
        }
    }

    // ── Phase 2: back substitution ────────────────────────────────────────────
    for iv in (1..nsys).rev() {
        let (d0, d1) = (bl.vdel[[2, 0, iv]], bl.vdel[[2, 1, iv]]);
        for kv in 0..iv {
            bl.vdel[[0, 0, kv]] -= bl.vm(0, iv, kv) * d0;
            bl.vdel[[1, 0, kv]] -= bl.vm(1, iv, kv) * d0;
            bl.vdel[[2, 0, kv]] -= bl.vm(2, iv, kv) * d0;
            bl.vdel[[0, 1, kv]] -= bl.vm(0, iv, kv) * d1;
            bl.vdel[[1, 1, kv]] -= bl.vm(1, iv, kv) * d1;
            bl.vdel[[2, 1, kv]] -= bl.vm(2, iv, kv) * d1;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_2x2() {
        // [2,1; 1,3] x = [5; 10] → x = [1, 3]
        let mut z = vec![2.0, 1.0, 1.0, 3.0];
        let mut r = vec![5.0, 10.0];
        gauss(2, &mut z, &mut r, 1);
        assert!((r[0] - 1.0).abs() < 1e-12, "x[0]={}", r[0]);
        assert!((r[1] - 3.0).abs() < 1e-12, "x[1]={}", r[1]);
    }

    #[test]
    fn gauss_3x3() {
        // [1,2,0; 3,4,0; 0,0,5] x = [5; 11; 10] → x = [1, 2, 2]
        let mut z = vec![1.0,2.0,0.0, 3.0,4.0,0.0, 0.0,0.0,5.0];
        let mut r = vec![5.0, 11.0, 10.0];
        gauss(3, &mut z, &mut r, 1);
        assert!((r[0] - 1.0).abs() < 1e-12, "x[0]={}", r[0]);
        assert!((r[1] - 2.0).abs() < 1e-12, "x[1]={}", r[1]);
        assert!((r[2] - 2.0).abs() < 1e-12, "x[2]={}", r[2]);
    }

    #[test]
    fn gauss4_identity_rhs() {
        let mut sys = [
            [1.0, 0.0, 0.0, 0.0, 7.0],
            [0.0, 1.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 1.0, 2.0],
        ];
        let sol = gauss4(&mut sys);
        assert!((sol[0] - 7.0).abs() < 1e-12);
        assert!((sol[1] - 3.0).abs() < 1e-12);
        assert!((sol[2] - 5.0).abs() < 1e-12);
        assert!((sol[3] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn ludcmp_baksub_roundtrip() {
        // [4,3; 6,3] x = [10; 12] → x = [1, 2]
        let mut a = vec![4.0, 3.0, 6.0, 3.0];
        let mut piv = vec![0i32; 2];
        ludcmp(&mut a, 2, 2, &mut piv);
        let mut b = vec![10.0, 12.0];
        baksub(&a, 2, 2, &piv, &mut b);
        assert!((b[0] - 1.0).abs() < 1e-12, "x[0]={}", b[0]);
        assert!((b[1] - 2.0).abs() < 1e-12, "x[1]={}", b[1]);
    }
}
