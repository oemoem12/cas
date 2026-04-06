#!/usr/bin/env bash
set -euo pipefail

# caS - catch AI models on your PC or Server
# Installation script

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[caS]${NC} $*"; }
warn()  { echo -e "${YELLOW}[caS]${NC} $*"; }
error() { echo -e "${RED}[caS]${NC} $*"; exit 1; }

# Check Python
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install Python 3.10+ first."
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PYTHON_VERSION found"

# Determine install method
USE_PIPX=false
if command -v pipx &>/dev/null; then
    USE_PIPX=true
fi

# Check if running as root
if [ "$(id -u)" -eq 0 ]; then
    warn "Running as root. Using --break-system-packages flag."
    PIP_FLAGS="--break-system-packages"
else
    PIP_FLAGS="--user --break-system-packages"
fi

# Install Python dependencies
info "Installing Python dependencies..."
pip install $PIP_FLAGS \
    fastapi \
    uvicorn[standard] \
    transformers \
    torch \
    accelerate \
    huggingface-hub \
    pydantic \
    modelscope \
    gguf \
    protobuf \
    sentencepiece \
    2>&1 | grep -v "^$" || true

# Install caS itself
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
info "Installing caS from $SCRIPT_DIR..."
pip install $PIP_FLAGS "$SCRIPT_DIR" --force-reinstall --no-deps 2>&1 | grep -v "^$" || true

# Verify installation
if command -v cas &>/dev/null; then
    info "caS installed successfully!"
    cas --help
else
    # Try adding ~/.local/bin to PATH
    LOCAL_BIN="$HOME/.local/bin"
    if [ -x "$LOCAL_BIN/cas" ]; then
        info "caS installed to $LOCAL_BIN/cas"
        if ! echo "$PATH" | grep -q "$LOCAL_BIN"; then
            warn "$LOCAL_BIN is not in your PATH."
            echo ""
            echo "Add this to your ~/.bashrc or ~/.zshrc:"
            echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
            echo ""
            echo "Or run directly: $LOCAL_BIN/cas --help"
        fi
    else
        error "caS installation failed. Please check the output above for errors."
    fi
fi

echo ""
info "Quick start:"
echo "  cas pull Qwen/Qwen2.5-0.5B --source hf-mirror"
echo "  cas pull bartowski/Qwen2.5-0.5B-Instruct-GGUF --gguf --quant Q4_K_M --source hf-mirror"
echo "  cas run Qwen/Qwen2.5-0.5B \"Hello!\""
echo "  cas chat bartowski/Qwen2.5-0.5B-Instruct-GGUF"
echo "  cas serve --port 8000"
echo ""
info "Done!"
