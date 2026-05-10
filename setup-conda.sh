#!/usr/bin/env bash
# =============================================================================
#  Pollen Analysis Tool — Conda Setup Script  (Linux & macOS)
#  https://github.com/Riha-Lab/Pollen-Analysis-Tool
#
#  Usage:
#    bash setup-conda.sh          # auto-detects GPU, installs everything
#    POLLEN_CPU_ONLY=1 bash setup-conda.sh   # force CPU-only install
#
#  After setup:
#    conda activate pollen-analysis
#    python pollen_analysis_app.py
# =============================================================================

set -euo pipefail

ENV_NAME="pollen-analysis"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/environment.yml"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
banner()  { echo -e "\n${BOLD}$*${RESET}\n"; }

# ── 0. Banner ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   🌸  Pollen Analysis Tool — Conda Setup         ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo ""

# ── 1. Check environment.yml exists ─────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    error "environment.yml not found in $SCRIPT_DIR"
    error "Make sure you cloned the full repository:"
    error "  git clone https://github.com/Riha-Lab/Pollen-Analysis-Tool.git"
    exit 1
fi

# ── 2. Locate conda ──────────────────────────────────────────────────────────
banner "Step 1 — Checking for conda"

_find_conda() {
    # Try obvious locations in order
    for candidate in \
        "$(command -v conda 2>/dev/null)" \
        "$HOME/miniforge3/bin/conda" \
        "$HOME/mambaforge/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/miniconda3/bin/conda" \
        "/opt/conda/bin/conda" \
        "/usr/local/bin/conda"
    do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

CONDA_EXE=""
if ! CONDA_EXE=$(_find_conda); then
    warn "conda not found — installing Miniforge (minimal conda-forge base)..."
    OS="$(uname -s)"
    ARCH="$(uname -m)"
    case "$OS-$ARCH" in
        Linux-x86_64)   INSTALLER="Miniforge3-Linux-x86_64.sh" ;;
        Linux-aarch64)  INSTALLER="Miniforge3-Linux-aarch64.sh" ;;
        Darwin-x86_64)  INSTALLER="Miniforge3-MacOSX-x86_64.sh" ;;
        Darwin-arm64)   INSTALLER="Miniforge3-MacOSX-arm64.sh" ;;
        *)
            error "Unsupported platform: $OS-$ARCH"
            error "Please install Miniforge manually from:"
            error "  https://github.com/conda-forge/miniforge/releases/latest"
            exit 1
            ;;
    esac
    TMP_INSTALLER="/tmp/$INSTALLER"
    info "Downloading $INSTALLER..."
    curl -fsSL \
        "https://github.com/conda-forge/miniforge/releases/latest/download/$INSTALLER" \
        -o "$TMP_INSTALLER"
    bash "$TMP_INSTALLER" -b -p "$HOME/miniforge3"
    rm -f "$TMP_INSTALLER"
    CONDA_EXE="$HOME/miniforge3/bin/conda"
    # Initialise for this shell session
    eval "$("$CONDA_EXE" shell.bash hook)"
    success "Miniforge installed at $HOME/miniforge3"
    info "Running 'conda init' so future terminals pick it up automatically..."
    "$CONDA_EXE" init bash 2>/dev/null || true
    [[ -f "$HOME/.zshrc" ]] && "$CONDA_EXE" init zsh 2>/dev/null || true
else
    # Activate base for this script session
    eval "$("$CONDA_EXE" shell.bash hook)" 2>/dev/null || true
    success "Found conda at: $CONDA_EXE"
fi

# ── 3. GPU detection ─────────────────────────────────────────────────────────
banner "Step 2 — Detecting hardware"

USE_GPU=false
if [[ "${POLLEN_CPU_ONLY:-0}" == "1" ]]; then
    info "POLLEN_CPU_ONLY=1 set — skipping GPU detection, installing CPU build."
elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown GPU")
    success "NVIDIA GPU detected: $GPU_NAME"
    info "Will install PyTorch with CUDA 12.1 support."
    info "(conda manages the CUDA runtime — no system CUDA install needed)"
    USE_GPU=true
else
    warn "No NVIDIA GPU detected — installing CPU-only PyTorch."
    warn "If you have a GPU, make sure the NVIDIA driver is installed,"
    warn "then re-run this script."
fi

# ── 4. Build the environment file to use ─────────────────────────────────────
banner "Step 3 — Preparing environment"

if [[ "$USE_GPU" == false ]]; then
    info "Patching environment.yml for CPU-only PyTorch..."
    TMP_ENV="/tmp/pollen_environment_cpu.yml"
    # Replace GPU-specific pytorch lines with CPU equivalents
    sed \
        's/- pytorch::pytorch-cuda=12\.1/- pytorch::cpuonly/' \
        "$ENV_FILE" > "$TMP_ENV"
    ACTIVE_ENV_FILE="$TMP_ENV"
    info "Using CPU environment file: $TMP_ENV"
else
    ACTIVE_ENV_FILE="$ENV_FILE"
fi

# ── 5. Create or update the environment ──────────────────────────────────────
banner "Step 4 — Installing packages"

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    info "Environment '${ENV_NAME}' already exists — updating to latest spec..."
    conda env update -n "$ENV_NAME" -f "$ACTIVE_ENV_FILE" --prune
    success "Environment updated."
else
    info "Creating new environment '${ENV_NAME}'..."
    info "This will download ~2–4 GB on first run. Please be patient."
    conda env create -n "$ENV_NAME" -f "$ACTIVE_ENV_FILE"
    success "Environment created."
fi

# Cleanup temp file if created
[[ -f "/tmp/pollen_environment_cpu.yml" ]] && rm -f /tmp/pollen_environment_cpu.yml

# ── 6. Verify the install ─────────────────────────────────────────────────────
banner "Step 5 — Verifying installation"

conda run -n "$ENV_NAME" python - <<'PYCHECK'
import sys
errors = []

# Core imports
for pkg in ["PyQt6", "numpy", "cv2", "matplotlib", "cellpose",
            "scipy", "statsmodels", "pandas", "reportlab",
            "PIL", "torch", "huggingface_hub", "requests"]:
    try:
        __import__(pkg)
    except ImportError as e:
        errors.append(f"  MISSING: {pkg} — {e}")

import torch
gpu_available = torch.cuda.is_available()

print(f"  Python:        {sys.version.split()[0]}")
print(f"  PyTorch:       {torch.__version__}")
print(f"  CUDA available:{' YES ✔' if gpu_available else ' no (CPU mode)'}")
print(f"  CUDA version:  {torch.version.cuda or 'N/A'}")

if errors:
    print("\nMissing packages:")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("\nAll packages verified ✔")
PYCHECK

success "All packages installed and verified."

# ── 7. macOS — check XQuartz for GUI display ─────────────────────────────────
if [[ "$(uname -s)" == "Darwin" ]]; then
    if ! command -v xquartz &>/dev/null && [[ ! -d "/Applications/Utilities/XQuartz.app" ]]; then
        echo ""
        warn "macOS detected: XQuartz is recommended for the best GUI experience."
        warn "Install from: https://www.xquartz.org/"
        warn "(The app may still work without it via native Qt rendering)"
    fi
fi

# ── 8. Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   ✅  Setup complete!                             ║${RESET}"
echo -e "${BOLD}╠══════════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║                                                  ║${RESET}"
echo -e "${BOLD}║   To launch the Pollen Analysis Tool:            ║${RESET}"
echo -e "${BOLD}║                                                  ║${RESET}"
echo -e "${BOLD}║     conda activate pollen-analysis               ║${RESET}"
echo -e "${BOLD}║     python pollen_analysis_app.py                ║${RESET}"
echo -e "${BOLD}║                                                  ║${RESET}"
echo -e "${BOLD}║   To update later:                               ║${RESET}"
echo -e "${BOLD}║     git pull                                     ║${RESET}"
echo -e "${BOLD}║     bash setup-conda.sh                          ║${RESET}"
echo -e "${BOLD}║                                                  ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo ""
