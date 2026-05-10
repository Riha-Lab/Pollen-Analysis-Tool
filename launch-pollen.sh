#!/usr/bin/env bash
# =============================================================================
#  Pollen Analysis Tool — Quick Launcher  (Linux & macOS)
#  Run setup-conda.sh first if you haven't already.
#  Usage:  bash launch-pollen.sh
# =============================================================================

ENV_NAME="pollen-analysis"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find conda
_find_conda() {
    for candidate in \
        "$(command -v conda 2>/dev/null)" \
        "$HOME/miniforge3/bin/conda" \
        "$HOME/mambaforge/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/miniconda3/bin/conda" \
        "/opt/conda/bin/conda"
    do
        [[ -x "$candidate" ]] && echo "$candidate" && return 0
    done
    return 1
}

CONDA_EXE=""
if ! CONDA_EXE=$(_find_conda); then
    echo "[ERROR] conda not found. Please run setup-conda.sh first."
    exit 1
fi

eval "$("$CONDA_EXE" shell.bash hook)" 2>/dev/null || true
conda activate "$ENV_NAME" 2>/dev/null || {
    echo "[ERROR] Environment '$ENV_NAME' not found. Please run setup-conda.sh first."
    exit 1
}

exec python "$SCRIPT_DIR/pollen_analysis_app.py"
