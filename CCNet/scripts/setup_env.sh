#!/usr/bin/env bash
# Quick helper to create a Python 3.10 virtual environment for CCNet.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${REPO_ROOT}/.venv-ccnet"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  cat <<'EOF' >&2
[setup_env] Python 3.10 is required but was not found.
Install it via:
  - Homebrew:     brew install python@3.10
  - or Pyenv:     pyenv install 3.10.14
Then re-run: PYTHON_BIN=python3.10 bash scripts/setup_env.sh
EOF
  exit 1
fi

echo "[setup_env] Creating virtual environment at ${VENV_PATH}"
${PYTHON_BIN} -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

pip install --upgrade pip setuptools wheel
pip install -r "${REPO_ROOT}/requirements.txt"

echo "[setup_env] Done. Activate via: source ${VENV_PATH}/bin/activate"

