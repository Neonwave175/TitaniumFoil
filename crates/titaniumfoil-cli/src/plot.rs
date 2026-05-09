// Braille terminal line plotter.
//
// Each Unicode braille character is a 2×4 pixel cell, giving twice the
// horizontal and 4× the vertical resolution of plain block characters.
//
// Braille dot layout (col × row → bit):
//   col 0: bits 0,1,2,3  (rows top→bottom)
//   col 1: bits 4,5,6,7

const BRAILLE: u32 = 0x2800;

// ANSI
const CYAN:  &str = "\x1b[38;5;117m";
const GREEN: &str = "\x1b[38;5;114m";
const DIM:   &str = "\x1b[38;5;240m";
const MUTED: &str = "\x1b[38;5;245m";
const WHITE: &str = "\x1b[38;5;253m";
const RESET: &str = "\x1b[0m";

// ── Public API ────────────────────────────────────────────────────────────────

pub struct Series<'a> {
    pub label: &'a str,
    pub color: &'a str,
    pub data:  &'a [(f64, f64)],  // (x, y)
}

/// Render a line plot into a String and print it.
pub fn print_plot(
    title:   &str,
    x_label: &str,
    _y_label: &str,
    series:  &[Series],
    cols:    usize,   // character columns (braille pixels = cols×2)
    rows:    usize,   // character rows    (braille pixels = rows×4)
) {
    let px_w = cols * 2;
    let px_h = rows * 4;

    // Range
    let xs: Vec<f64> = series.iter().flat_map(|s| s.data.iter().map(|p| p.0)).collect();
    let ys: Vec<f64> = series.iter().flat_map(|s| s.data.iter().map(|p| p.1)).collect();
    if xs.is_empty() { return; }

    let x_min = xs.iter().cloned().fold(f64::INFINITY,     f64::min);
    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = ys.iter().cloned().fold(f64::INFINITY,     f64::min);
    let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (x_max - x_min).abs() < 1e-12 || (y_max - y_min).abs() < 1e-12 { return; }

    // Pixel grid: stores series index + 1 (0 = empty)
    let mut grid = vec![0u8; px_w * px_h];

    for (si, s) in series.iter().enumerate() {
        let color_id = (si + 1) as u8;
        for w in s.data.windows(2) {
            let (x0, y0) = w[0];
            let (x1, y1) = w[1];
            let px0 = to_px(x0, x_min, x_max, px_w);
            let py0 = px_h - 1 - to_px(y0, y_min, y_max, px_h);
            let px1 = to_px(x1, x_min, x_max, px_w);
            let py1 = px_h - 1 - to_px(y1, y_min, y_max, px_h);
            draw_line(&mut grid, px_w, px_h, px0, py0, px1, py1, color_id);
        }
    }

    // Y-axis label width
    let ax_w = 7usize;

    // Title
    println!("{}{MUTED}{title}{RESET}", " ".repeat(ax_w));

    // Rows
    for row in 0..rows {
        // Y-axis tick at top, middle, bottom
        let py = row * 4;
        let y_val = y_max - (py as f64 / (px_h - 1) as f64) * (y_max - y_min);
        if row == 0 || row == rows / 2 || row == rows - 1 {
            print!("{DIM}{:>6.2}{RESET} ", y_val);
        } else if row == rows / 4 || row == 3 * rows / 4 {
            print!("{DIM}{:>6.2}{RESET} ", y_val);
        } else {
            print!("{:ax_w$}", "");
        }

        for col in 0..cols {
            let mut bits = 0u8;
            let mut cid  = 0u8;
            for br in 0..4usize {
                for bc in 0..2usize {
                    let px = col * 2 + bc;
                    let py = row * 4 + br;
                    if px < px_w && py < px_h {
                        let v = grid[py * px_w + px];
                        if v != 0 {
                            bits |= 1 << (bc * 4 + br);
                            cid = v;
                        }
                    }
                }
            }
            let ch = char::from_u32(BRAILLE + bits as u32).unwrap_or(' ');
            if cid > 0 {
                let color = series[cid as usize - 1].color;
                print!("{color}{ch}{RESET}");
            } else {
                print!("{DIM}·{RESET}");
            }
        }
        println!();
    }

    // X-axis ticks
    let tick_mid = format!("{:.2}", (x_min + x_max) / 2.0);
    println!(
        "{:ax_w$}{DIM}{:<width$.2}  {:^mw$}  {:>width$.2}{RESET}",
        "", x_min, tick_mid, x_max,
        width = (cols - tick_mid.len()) / 3,
        mw    = tick_mid.len(),
    );
    println!("{:ax_w$}{DIM}{:^cols$}{RESET}", "", x_label);

    // Legend
    for s in series {
        println!("{:ax_w$}{}{} ─ {}{RESET}", "", s.color, "▪", s.label);
    }
    println!();
}

// ── Two side-by-side plots ────────────────────────────────────────────────────

pub fn print_polar(
    naca:   &str,
    re:     f64,
    alphas: &[f64],
    cls:    &[f64],
    lds:    &[f64],
    cps:    &[f64],   // Cp at ~25% chord (suction peak proxy)
) {
    println!();
    println!("  {WHITE}NACA {naca}   Re={re:.0}{RESET}");
    println!("  {DIM}{}{RESET}", "─".repeat(70));

    let xy_cl: Vec<(f64, f64)> = alphas.iter().zip(cls.iter()).map(|(&a, &c)| (a, c)).collect();
    let xy_ld: Vec<(f64, f64)> = alphas.iter().zip(lds.iter()).map(|(&a, &l)| (a, l)).collect();
    let xy_cp: Vec<(f64, f64)> = alphas.iter().zip(cps.iter()).map(|(&a, &p)| (a, p)).collect();

    print_plot(
        "lift curve  CL vs α",
        "α (°)", "CL",
        &[Series { label: "CL",  color: CYAN,  data: &xy_cl }],
        52, 18,
    );

    print_plot(
        "efficiency  L/D vs α",
        "α (°)", "L/D",
        &[Series { label: "L/D", color: GREEN, data: &xy_ld }],
        52, 18,
    );

    print_plot(
        "suction peak  −Cp(25%) vs α",
        "α (°)", "−Cp",
        &[Series { label: "−Cp @ 25%", color: MUTED, data: &xy_cp }],
        52, 12,
    );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn to_px(v: f64, lo: f64, hi: f64, n: usize) -> usize {
    ((v - lo) / (hi - lo) * (n as f64 - 1.0))
        .round()
        .clamp(0.0, (n - 1) as f64) as usize
}

fn draw_line(
    grid: &mut [u8], w: usize, h: usize,
    x0: usize, y0: usize, x1: usize, y1: usize,
    color: u8,
) {
    let (mut x, mut y) = (x0 as isize, y0 as isize);
    let dx = (x1 as isize - x).abs();
    let dy = (y1 as isize - y).abs();
    let sx: isize = if x0 < x1 { 1 } else { -1 };
    let sy: isize = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    loop {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            grid[y as usize * w + x as usize] = color;
        }
        if x == x1 as isize && y == y1 as isize { break; }
        let e2 = 2 * err;
        if e2 > -dy { err -= dy; x += sx; }
        if e2 <  dx { err += dx; y += sy; }
    }
}
