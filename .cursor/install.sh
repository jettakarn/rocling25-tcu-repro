#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the ROCLING-2025 TCU reproduction.
# Prepares a CPU-only Python environment, clones the shared-task dataset, and
# normalizes it into data/processed/. Heavy steps (encoder download, embedding,
# training) are left to the documented run commands in README.md.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

# 1. Ensure venv support (ensurepip ships in python3.x-venv on Debian/Ubuntu).
if ! "$PYTHON_BIN" -c "import ensurepip" >/dev/null 2>&1; then
  PYVER="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  sudo apt-get update -qq
  sudo apt-get install -y -qq "python${PYVER}-venv"
fi

# 2. Create the virtualenv once.
if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
VENV_PY="${VENV_DIR}/bin/python"

"$VENV_PY" -m pip install --upgrade pip setuptools wheel

# 3. Install a CPU-only torch build first so requirements.txt does not pull the
#    large CUDA wheels, then the remaining pinned dependencies.
"$VENV_PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$VENV_PY" -m pip install -r requirements.txt

# 4. Fetch the shared-task dataset if it is not already present.
DATA_REPO="data/raw/ROCLING-2025-ST-DSA-MST"
if [ ! -d "$DATA_REPO/Dataset" ]; then
  git clone --depth 1 https://github.com/NYCU-NLP/ROCLING-2025-ST-DSA-MST.git "$DATA_REPO"
fi

# 5. Normalize raw CSVs into data/processed/{train,dev,test}.csv.
if [ ! -f "data/processed/train.csv" ]; then
  "$VENV_PY" -m src.prepare_data
fi

echo "install.sh: environment ready."
