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
    // Strip env vars that maturin/PyO3 set and that can confuse xcrun
    // (SDKROOT, TARGET, MACOSX_DEPLOYMENT_TARGET, etc.)
    let path = std::env::var("PATH").unwrap_or_default();
    let home = std::env::var("HOME").unwrap_or_default();
    let air_out = Command::new("xcrun")
        .args(["metal", "-c", src.to_str().unwrap(), "-o", &air])
        .env_clear()
        .env("PATH", &path)
        .env("HOME", &home)
        .output();
    let air_ok = air_out.as_ref().map(|o| o.status.success()).unwrap_or(false);

    if !air_ok {
        if let Ok(ref o) = air_out {
            println!("cargo:warning=xcrun metal stderr: {}", String::from_utf8_lossy(&o.stderr).replace('\n', " | "));
            println!("cargo:warning=xcrun metal stdout: {}", String::from_utf8_lossy(&o.stdout).replace('\n', " | "));
        }
        println!("cargo:warning=metal compile failed for {name}.metal");
        return false;
    }

    // .air → .metallib
    let lib_ok = Command::new("xcrun")
        .args(["metallib", &air, "-o", &lib])
        .env_clear()
        .env("PATH", &path)
        .env("HOME", &home)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !lib_ok {
        eprintln!("build.rs: metallib link failed for {name}");
        return false;
    }

    true
}

/// If xcrun fails, look for a pre-compiled .metallib from another build of
/// the same crate (e.g. the workspace release build) in sibling OUT_DIR dirs.
fn find_precompiled(name: &str, out_dir: &str) -> Option<PathBuf> {
    // out_dir = .../target/<profile>/build/titaniumfoil-metal-HASH/out
    let out_path   = PathBuf::from(out_dir);
    let build_dir  = out_path.parent()?.parent()?; // .../target/<profile>/build/
    for entry in std::fs::read_dir(build_dir).ok()?.flatten() {
        let fname = entry.file_name().to_string_lossy().into_owned();
        if fname.starts_with("titaniumfoil-metal-") {
            let metallib = entry.path().join("out").join(format!("{name}.metallib"));
            if metallib.exists() {
                return Some(metallib);
            }
        }
    }
    None
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

    println!("cargo:rerun-if-changed=build.rs");

    let shaders = ["panel_influence", "blvar_compute", "blsys_solve"];
    let mut built = 0;

    for name in &shaders {
        let lib = format!("{out_dir}/{name}.metallib");
        // Already compiled into this OUT_DIR (incremental build)
        if Path::new(&lib).exists() {
            built += 1;
            continue;
        }

        if compile_shader(name, &shader_dir, &out_dir) {
            built += 1;
            println!("cargo:warning=Compiled {name}.metal → {name}.metallib");
        } else {
            // xcrun failed — try a pre-compiled metallib from another build
            if let Some(src) = find_precompiled(name, &out_dir) {
                if std::fs::copy(&src, &lib).is_ok() {
                    built += 1;
                    println!("cargo:warning=Copied pre-compiled {name}.metallib from {}", src.display());
                    continue;
                }
            }
            println!("cargo:warning=Skipped {name}.metal (xcrun failed, no pre-compiled fallback)");
        }
    }

    println!("cargo:rustc-env=METAL_SHADER_DIR={out_dir}");
    println!("cargo:warning=Metal shaders built: {built}/{}", shaders.len());
}
