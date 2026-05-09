// Compile Metal shaders to .metallib at build time via xcrun.

use std::process::Command;
use std::path::{Path, PathBuf};
use std::env;

fn compile_shader(name: &str, shader_dir: &Path, out_dir: &str) -> bool {
    let src = shader_dir.join(format!("{name}.metal"));
    let air = format!("{out_dir}/{name}.air");
    let lib = format!("{out_dir}/{name}.metallib");

    // Tell Cargo to re-run this build script whenever the .metal file changes.
    // MUST be printed before any early returns so Cargo always watches the file,
    // even if this compilation fails — otherwise Cargo forgets the file exists.
    println!("cargo:rerun-if-changed={}", src.display());

    if !src.exists() {
        eprintln!("build.rs: shader not found: {}", src.display());
        return false;
    }

    // .metal → .air
    let air_ok = Command::new("xcrun")
        .args(["metal", "-c", src.to_str().unwrap(), "-o", &air])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !air_ok {
        eprintln!("build.rs: metal compile failed for {name}.metal");
        return false;
    }

    // .air → .metallib
    let lib_ok = Command::new("xcrun")
        .args(["metallib", &air, "-o", &lib])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !lib_ok {
        eprintln!("build.rs: metallib link failed for {name}");
        return false;
    }

    true
}

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        println!("cargo:warning=Not macOS — Metal shaders skipped");
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir   = manifest_dir.join("../../shaders");
    let out_dir      = env::var("OUT_DIR").unwrap();

    // Also watch build.rs itself
    println!("cargo:rerun-if-changed=build.rs");

    let shaders = ["panel_influence", "blvar_compute", "blsys_solve"];
    let mut built = 0;

    for name in &shaders {
        if compile_shader(name, &shader_dir, &out_dir) {
            built += 1;
            println!("cargo:warning=Compiled {name}.metal → {name}.metallib");
        } else {
            println!("cargo:warning=Skipped {name}.metal (compile error — check shader syntax)");
        }
    }

    println!("cargo:rustc-env=METAL_SHADER_DIR={out_dir}");
    println!("cargo:warning=Metal shaders built: {built}/{}", shaders.len());
}
