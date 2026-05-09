// re_calc — Reynolds number calculator for titaniumfoil-optimizer
//
// Computes Re = V·c / ν  using ISA atmosphere.
// Handles any altitude (0–11 000 m troposphere), any speed unit, any chord unit.
//
// Usage:  titaniumfoil-re

use std::io::{self, Write};

// ── ANSI ─────────────────────────────────────────────────────────────────────
const WHITE:  &str = "\x1b[38;5;253m";
const CYAN:   &str = "\x1b[38;5;117m";
const GREEN:  &str = "\x1b[38;5;114m";
const YELLOW: &str = "\x1b[38;5;221m";
const DIM:    &str = "\x1b[38;5;240m";
const MUTED:  &str = "\x1b[38;5;245m";
const RESET:  &str = "\x1b[0m";
const BOLD:   &str = "\x1b[1m";

// ── ISA atmosphere ────────────────────────────────────────────────────────────
//
// Valid for 0 – 11 000 m (troposphere).
// Above 11 000 m the lapse rate is 0 (isothermal layer); extrapolation is
// clamped to that layer for altitudes up to 20 000 m.
//
// References:
//   ICAO Doc 7488 / ISO 2533
//   Sutherland (1893) viscosity law

const T0:    f64 = 288.15;     // K   sea-level temperature
const P0:    f64 = 101_325.0;  // Pa  sea-level pressure
const L:     f64 = 0.006_5;    // K/m tropospheric lapse rate
const G:     f64 = 9.806_65;   // m/s²
const R_AIR: f64 = 287.058;    // J/(kg·K)
const EXP:   f64 = G / (R_AIR * L);   // ≈ 5.2561

// Sutherland constants for air
const MU_REF: f64 = 1.716e-5;  // Pa·s  at T_REF
const T_REF:  f64 = 273.15;    // K
const S:      f64 = 110.4;     // K  Sutherland constant

fn isa(altitude_m: f64) -> (f64, f64, f64) {
    // returns (temperature K, density kg/m³, kinematic viscosity m²/s)
    let h = altitude_m.clamp(0.0, 20_000.0);

    let (temp, pressure) = if h <= 11_000.0 {
        let t = T0 - L * h;
        let p = P0 * (t / T0).powf(EXP);
        (t, p)
    } else {
        // isothermal layer 11 000 – 20 000 m
        let t11 = T0 - L * 11_000.0;  // 216.65 K
        let p11 = P0 * (t11 / T0).powf(EXP);
        let p   = p11 * (-(G / (R_AIR * t11)) * (h - 11_000.0)).exp();
        (t11, p)
    };

    let rho = pressure / (R_AIR * temp);

    // Sutherland dynamic viscosity
    let mu = MU_REF * (temp / T_REF).powf(1.5) * (T_REF + S) / (temp + S);

    let nu = mu / rho;
    (temp, rho, nu)
}

// ── I/O helpers ───────────────────────────────────────────────────────────────

fn prompt(msg: &str) -> String {
    print!("{}", msg);
    io::stdout().flush().unwrap();
    let mut buf = String::new();
    io::stdin().read_line(&mut buf).unwrap();
    buf.trim().to_string()
}

fn parse_f(s: &str, default: f64) -> f64 {
    if s.is_empty() { default } else { s.parse().unwrap_or(default) }
}

fn sep() {
    println!("{DIM}{}{RESET}", "─".repeat(62));
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!();
    sep();
    println!("  {WHITE}{BOLD}reynolds number calculator{RESET}  \
              {DIM}ISA atmosphere · Sutherland viscosity{RESET}");
    sep();
    println!();

    // ── Altitude ──────────────────────────────────────────────────────────────
    println!("{MUTED}altitude{RESET}  {DIM}(default 0 m sea level){RESET}");
    let alt_m = parse_f(
        &prompt(&format!("  {DIM}altitude [m]  >{RESET} ")),
        0.0,
    );

    let (temp_k, rho, nu) = isa(alt_m);
    let temp_c = temp_k - 273.15;
    let a_ms   = (1.4 * R_AIR * temp_k).sqrt();   // speed of sound m/s

    println!("  {DIM}T = {temp_c:.1}°C   ρ = {rho:.4} kg/m³   \
              ν = {:.3e} m²/s   a = {a_ms:.1} m/s{RESET}\n",
             nu);

    // ── Chord ─────────────────────────────────────────────────────────────────
    println!("{MUTED}chord length{RESET}");
    let chord_raw = parse_f(
        &prompt(&format!("  {DIM}chord value  >{RESET} ")),
        100.0,
    );
    let chord_unit = prompt(&format!("  {DIM}unit [mm/cm/m/in]  (default mm)  >{RESET} "));
    let chord_m = match chord_unit.to_lowercase().trim() {
        "m"  => chord_raw,
        "cm" => chord_raw / 100.0,
        "in" => chord_raw * 0.0254,
        _    => chord_raw / 1000.0,   // default mm
    };
    println!("  {DIM}chord = {chord_m:.4} m{RESET}\n");

    // ── Speed (one or more) ───────────────────────────────────────────────────
    println!("{MUTED}airspeed{RESET}  {DIM}enter one or more values separated by spaces{RESET}");
    let speed_raw_str = prompt(&format!("  {DIM}speed values  >{RESET} "));
    let speed_unit    = prompt(&format!("  {DIM}unit [m/s / km/h / mph / kts]  (default m/s)  >{RESET} "));

    let to_ms: f64 = match speed_unit.to_lowercase().trim() {
        "km/h" | "kmh" | "kph" => 1.0 / 3.6,
        "mph"                   => 0.44704,
        "kts" | "kt" | "knots" => 0.51444,
        _                       => 1.0,   // m/s
    };

    let speeds_ms: Vec<f64> = speed_raw_str
        .split_whitespace()
        .filter_map(|s| s.parse::<f64>().ok())
        .map(|v| v * to_ms)
        .collect();

    if speeds_ms.is_empty() {
        println!("{DIM}no valid speeds entered — using 10 m/s{RESET}");
    }
    let speeds_ms = if speeds_ms.is_empty() { vec![10.0] } else { speeds_ms };

    println!();

    // ── Results ───────────────────────────────────────────────────────────────
    sep();
    println!("  {WHITE}{BOLD}reynolds numbers{RESET}  \
              {DIM}chord={chord_m:.4}m  alt={alt_m:.0}m{RESET}");
    sep();

    let mut re_list: Vec<u64> = Vec::new();

    for &v_ms in &speeds_ms {
        let re  = v_ms * chord_m / nu;
        let ma  = v_ms / a_ms;
        let re_k = re / 1000.0;

        let re_color = if re < 50_000.0 { YELLOW }
                       else if re < 500_000.0 { GREEN }
                       else { CYAN };

        println!(
            "  {MUTED}{:>8.2} {unit:<4}{RESET}  {DIM}({v_ms:>6.2} m/s  Ma {ma:.4}){RESET}  \
             {re_color}{BOLD}Re = {re:.0}{RESET}  {DIM}({re_k:.1}k){RESET}",
            v_ms / to_ms,
            unit = speed_unit.trim(),
        );

        re_list.push(re.round() as u64);
    }

    println!();

    // ── Regime notes ─────────────────────────────────────────────────────────
    let any_re = re_list[0] as f64;
    if any_re < 30_000.0 {
        println!("  {YELLOW}⚠  Re < 30k — strongly laminar, separation likely{RESET}");
    } else if any_re < 100_000.0 {
        println!("  {YELLOW}   Re 30k–100k — transitional; viscous effects dominate{RESET}");
    } else if any_re < 500_000.0 {
        println!("  {GREEN}   Re 100k–500k — typical UAV / model aircraft range{RESET}");
    } else {
        println!("  {CYAN}   Re > 500k — approaching full-scale aviation{RESET}");
    }

    println!();

    // ── Ready-to-paste list ────────────────────────────────────────────────────
    let paste: Vec<String> = re_list.iter().map(|r| r.to_string()).collect();
    println!("  {MUTED}paste into titaniumfoil-opt prompt [1]{RESET}");
    println!("  {WHITE}{BOLD}{}{RESET}\n", paste.join(", "));
}
