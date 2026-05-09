#!/usr/bin/env bash
set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $*${NC}"; }
info() { echo -e "${CYAN}  → $*${NC}"; }
die()  { echo -e "${RED}  ✗ $*${NC}"; exit 1; }

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║      TitaniumFoil  build script      ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${NC}"

# ── macOS check ───────────────────────────────────────────────────────────────
[[ "$(uname)" == "Darwin" ]] || die "This project requires macOS (Metal GPU support)"

# ── Xcode command-line tools ──────────────────────────────────────────────────
info "Checking Xcode command-line tools..."
if ! xcode-select -p &>/dev/null; then
    warn "Xcode CLT not found — installing..."
    xcode-select --install
    echo "  Re-run this script after the Xcode installer finishes."
    exit 0
fi
ok "Xcode CLT: $(xcode-select -p)"

# ── Metal toolchain ───────────────────────────────────────────────────────────
info "Checking Metal shader compiler..."
if ! xcrun metal --version &>/dev/null 2>&1; then
    warn "Metal toolchain missing — downloading (~700 MB)..."
    xcodebuild -downloadComponent MetalToolchain
fi
METAL_VER=$(xcrun metal --version 2>&1 | head -1)
ok "Metal: $METAL_VER"

# ── Homebrew ──────────────────────────────────────────────────────────────────
info "Checking Homebrew..."
if ! command -v brew &>/dev/null; then
    warn "Homebrew not found — installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
ok "Homebrew: $(brew --version | head -1)"

# ── Rust ─────────────────────────────────────────────────────────────────────
info "Checking Rust..."
if ! command -v rustc &>/dev/null && [[ ! -f "$HOME/.cargo/env" ]]; then
    warn "Rust not found — installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
fi

# source cargo env if needed
[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"

if ! command -v rustc &>/dev/null; then
    die "Rust still not found after install — open a new terminal and re-run"
fi

RUST_INSTALLED=$(rustc --version)
info "Updating Rust toolchain..."
rustup update stable --no-self-update 2>&1 | grep -E "updated|unchanged|stable" | head -2
RUST_CURRENT=$(rustc --version)
ok "Rust: $RUST_CURRENT"

# ── Check Rust version (need 1.70+ for edition 2021 features) ─────────────────
RUST_MINOR=$(rustc --version | grep -oE '[0-9]+\.[0-9]+' | head -1 | cut -d. -f2)
if [[ "$RUST_MINOR" -lt 70 ]]; then
    die "Rust 1.70+ required (have $(rustc --version)) — run: rustup update stable"
fi

# ── Build ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
info "Building TitaniumFoil (release)..."
echo ""

cargo build --release 2>&1

echo ""
ok "Build complete!"
echo ""
echo -e "${CYAN}  Binaries:${NC}"
echo "    ./target/release/titaniumfoil        — solver"
echo "    ./target/release/titaniumfoil-bench  — GPU vs CPU benchmark"
echo ""
echo -e "${CYAN}  Usage:${NC}"
echo "    ./target/release/titaniumfoil 0012 5"
echo "    ./target/release/titaniumfoil 4412 aseq -5 15 1"
echo "    ./target/release/titaniumfoil 0012 5 1e6 180      # N=359 panels"
echo "    ./target/release/titaniumfoil-bench 0012 5"
echo ""
