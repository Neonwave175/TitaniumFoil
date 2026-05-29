#!/usr/bin/env bash
# release.sh — build and publish a GitHub release with pre-compiled binaries.
#
# Usage:
#   ./release.sh              # uses 'gh' CLI if installed
#   GITHUB_TOKEN=xxx ./release.sh   # uses curl with a PAT
#
# Requires: cargo, maturin, and either 'gh' or a GITHUB_TOKEN env var.

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
info() { echo -e "${CYAN}  → $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $*${NC}"; }
die()  { echo -e "${RED}  ✗ $*${NC}"; exit 1; }

REPO="Neonwave175/TitaniumFoil"
VERSION="v0.1.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║   TitaniumFoil release packager      ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${NC}"

# ── Build ─────────────────────────────────────────────────────────────────────
[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"

info "Building Rust CLI + FFI..."
cargo build --release -p titaniumfoil -p titaniumfoil-cli \
    -p titaniumfoil-optimizer -p titaniumfoil-ffi 2>&1 | grep -E "Compiling titan|Finished"

info "Building Python wheel..."
maturin build --release \
    --manifest-path crates/titaniumfoil-py/Cargo.toml 2>&1 | grep -E "Built wheel|Finished"

ok "All binaries built."

# ── Package ───────────────────────────────────────────────────────────────────
DIST="$(mktemp -d)"
info "Packaging into $DIST ..."

# 1. CLI tarball
tar -czf "$DIST/titaniumfoil-$VERSION-macos-arm64.tar.gz" \
    -C target/release \
    titaniumfoil titaniumfoil-opt titaniumfoil-re titaniumfoil-bench
ok "CLI tarball: titaniumfoil-$VERSION-macos-arm64.tar.gz ($(du -sh "$DIST/titaniumfoil-$VERSION-macos-arm64.tar.gz" | cut -f1))"

# 2. C++ library tarball
CPP_TMP="$(mktemp -d)"
cp target/release/libtitaniumfoil.dylib \
   target/release/libtitaniumfoil.a "$CPP_TMP/"
cp -r include "$CPP_TMP/"
tar -czf "$DIST/titaniumfoil-cpp-$VERSION-macos-arm64.tar.gz" -C "$CPP_TMP" .
ok "C++ tarball: titaniumfoil-cpp-$VERSION-macos-arm64.tar.gz ($(du -sh "$DIST/titaniumfoil-cpp-$VERSION-macos-arm64.tar.gz" | cut -f1))"

# 3. Python wheel
WHL="$(ls target/wheels/titaniumfoil-*.whl | head -1)"
cp "$WHL" "$DIST/"
ok "Python wheel: $(basename "$WHL") ($(du -sh "$WHL" | cut -f1))"

echo ""
info "Assets ready:"
ls -lh "$DIST/"

# ── Publish ───────────────────────────────────────────────────────────────────
echo ""

RELEASE_NOTES="## TitaniumFoil $VERSION

**Apple Silicon only (M1 / M2 / M3 / M4)**

### CLI binaries
\`\`\`bash
tar -xzf titaniumfoil-$VERSION-macos-arm64.tar.gz
./titaniumfoil 4412 aseq -5 15 1 200000
./titaniumfoil-opt
\`\`\`

### Python (any Python 3.8+)
\`\`\`bash
pip install titaniumfoil-0.1.0-cp38-abi3-macosx_11_0_arm64.whl
\`\`\`

### C++ library
Extract \`titaniumfoil-cpp-$VERSION-macos-arm64.tar.gz\` — includes \`libtitaniumfoil.dylib\`, \`libtitaniumfoil.a\`, and \`include/titaniumfoil.hpp\`.

### Build from source
\`\`\`bash
git clone https://github.com/$REPO.git
cd TitaniumFoil && ./build.sh
\`\`\`"

if command -v gh &>/dev/null; then
    info "Publishing via 'gh'..."
    gh release create "$VERSION" \
        --repo "$REPO" \
        --title "$VERSION — Initial Release" \
        --notes "$RELEASE_NOTES" \
        "$DIST"/titaniumfoil-"$VERSION"-macos-arm64.tar.gz \
        "$DIST"/titaniumfoil-cpp-"$VERSION"-macos-arm64.tar.gz \
        "$DIST"/titaniumfoil-*.whl
    ok "Release published: https://github.com/$REPO/releases/tag/$VERSION"

elif [[ -n "${GITHUB_TOKEN:-}" ]]; then
    info "Publishing via GitHub API (curl)..."

    # Create release
    RELEASE_JSON=$(curl -sf -X POST \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Content-Type: application/json" \
        "https://api.github.com/repos/$REPO/releases" \
        -d "{\"tag_name\":\"$VERSION\",\"name\":\"$VERSION — Initial Release\",\"body\":$(echo "$RELEASE_NOTES" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),\"draft\":false,\"prerelease\":false}")

    UPLOAD_URL=$(echo "$RELEASE_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['upload_url'])" | sed 's/{.*//')
    RELEASE_URL=$(echo "$RELEASE_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['html_url'])")

    upload() {
        local file="$1" name="$(basename "$1")"
        info "Uploading $name..."
        curl -sf -X POST \
            -H "Authorization: token $GITHUB_TOKEN" \
            -H "Content-Type: application/octet-stream" \
            "${UPLOAD_URL}?name=${name}" \
            --data-binary @"$file" > /dev/null
        ok "Uploaded: $name"
    }

    for f in "$DIST"/*; do upload "$f"; done
    ok "Release published: $RELEASE_URL"

else
    warn "Neither 'gh' nor GITHUB_TOKEN found."
    echo ""
    echo "  Option 1 — install gh and authenticate:"
    echo "    brew install gh && gh auth login"
    echo "    ./release.sh"
    echo ""
    echo "  Option 2 — set a Personal Access Token:"
    echo "    GITHUB_TOKEN=ghp_xxx ./release.sh"
    echo ""
    echo "  Option 3 — upload manually at:"
    echo "    https://github.com/$REPO/releases/new"
    echo "    Tag: $VERSION"
    echo "    Assets to upload (in $DIST):"
    ls "$DIST/"
fi
