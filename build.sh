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

# ── parse flags ───────────────────────────────────────────────────────────────
BUILD_PYTHON=0
PYTHON_VENV=""
for arg in "$@"; do
    case "$arg" in
        --python)          BUILD_PYTHON=1 ;;
        --venv=*)          PYTHON_VENV="${arg#--venv=}"; BUILD_PYTHON=1 ;;
        --help|-h)
            echo "Usage: ./build.sh [--python] [--venv=/path/to/venv]"
            echo ""
            echo "  (no flags)          build Rust CLI binaries only"
            echo "  --python            also build & install Python bindings"
            echo "  --venv=/path/venv   install Python bindings into that venv"
            echo ""
            exit 0 ;;
    esac
done

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
    warn "Metal toolchain missing — attempting to install..."

    # Xcode needs runFirstLaunch before component downloads work
    info "Running xcodebuild -runFirstLaunch to repair Xcode plugins..."
    sudo xcodebuild -runFirstLaunch 2>/dev/null || true

    info "Downloading Metal toolchain (~700 MB)..."
    if ! xcodebuild -downloadComponent MetalToolchain 2>/dev/null; then
        warn "Automatic download failed."
        echo ""
        echo "  Install the Metal toolchain manually:"
        echo "    1. Open Xcode → Settings → Platforms"
        echo "    2. Or: sudo xcodebuild -runFirstLaunch"
        echo "    3. Then re-run this script"
        echo ""
        warn "Continuing without Metal toolchain — shader compile will use cache if available."
    fi
fi

if xcrun metal --version &>/dev/null 2>&1; then
    ok "Metal: $(xcrun metal --version 2>&1 | head -1)"
else
    warn "Metal shader compiler not available — pre-compiled shaders will be used if present"
fi

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

[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"

if ! command -v rustc &>/dev/null; then
    die "Rust still not found after install — open a new terminal and re-run"
fi

info "Updating Rust toolchain..."
rustup update stable --no-self-update 2>&1 | grep -E "updated|unchanged|stable" | head -2
ok "Rust: $(rustc --version)"

RUST_MINOR=$(rustc --version | grep -oE '[0-9]+\.[0-9]+' | head -1 | cut -d. -f2)
[[ "$RUST_MINOR" -ge 70 ]] || die "Rust 1.70+ required — run: rustup update stable"

# ── Python / maturin (only when --python is set) ─────────────────────────────
if [[ "$BUILD_PYTHON" -eq 1 ]]; then
    info "Checking Python bindings prerequisites..."

    # Need Python 3.8+
    PYTHON_BIN=""
    for py in python3 python; do
        if command -v "$py" &>/dev/null; then
            PY_MAJOR=$("$py" -c "import sys; print(sys.version_info.major)")
            PY_MINOR=$("$py" -c "import sys; print(sys.version_info.minor)")
            if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 8 ]]; then
                PYTHON_BIN="$py"
                break
            fi
        fi
    done
    [[ -n "$PYTHON_BIN" ]] || die "Python 3.8+ not found — install Python then re-run"
    ok "Python: $($PYTHON_BIN --version)"

    # If a venv path was given, use its Python
    if [[ -n "$PYTHON_VENV" ]]; then
        [[ -d "$PYTHON_VENV" ]] || die "venv not found: $PYTHON_VENV  (create it first with: python3 -m venv $PYTHON_VENV)"
        INTERP_FLAG="-i $PYTHON_VENV/bin/python"
        ok "Target venv: $PYTHON_VENV"
    else
        INTERP_FLAG=""
        warn "No --venv specified — installing into the active Python environment"
    fi

    # Install maturin if missing
    info "Checking maturin..."
    if ! command -v maturin &>/dev/null; then
        if ! $PYTHON_BIN -c "import maturin" &>/dev/null 2>&1; then
            warn "maturin not found — installing via pip..."
            $PYTHON_BIN -m pip install --quiet maturin
        fi
    fi
    MATURIN_BIN="maturin"
    command -v maturin &>/dev/null || MATURIN_BIN="$PYTHON_BIN -m maturin"
    ok "maturin: $($MATURIN_BIN --version 2>&1 | head -1)"
fi

# ── Build Rust CLI ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
info "Building TitaniumFoil CLI (release)..."
echo ""
cargo build --release 2>&1
echo ""
ok "Rust build complete!"

# ── Build Python bindings ─────────────────────────────────────────────────────
if [[ "$BUILD_PYTHON" -eq 1 ]]; then
    echo ""
    info "Building Python bindings..."
    echo ""
    # shellcheck disable=SC2086
    $MATURIN_BIN develop --release \
        --manifest-path crates/titaniumfoil-py/Cargo.toml \
        $INTERP_FLAG
    echo ""
    ok "Python bindings installed!"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}  Binaries:${NC}"
echo "    ./target/release/titaniumfoil         — airfoil solver"
echo "    ./target/release/titaniumfoil-opt     — optimizer"
echo "    ./target/release/titaniumfoil-re      — Reynolds calculator"
echo "    ./target/release/titaniumfoil-bench   — benchmark"
echo ""
echo -e "${CYAN}  Usage:${NC}"
echo "    ./target/release/titaniumfoil 4412 aseq -5 15 1 200000"
echo "    ./target/release/titaniumfoil-opt"
echo "    ./target/release/titaniumfoil-re"
echo ""
if [[ "$BUILD_PYTHON" -eq 1 ]]; then
    echo -e "${CYAN}  Python:${NC}"
    if [[ -n "$PYTHON_VENV" ]]; then
        echo "    source $PYTHON_VENV/bin/activate"
    fi
    echo "    python3 -c \"from titaniumfoil import Solver; print(Solver().analyze('4412', 4.0, 200000))\""
    echo ""
fi
