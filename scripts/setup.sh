#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
PYTHON="$ROOT/.venv/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install torch==2.5.1
"$PYTHON" -m pip install -r requirements-lock.txt
"$PYTHON" -m pip install -e .
if [[ ! -d .external/circuit_sparsity ]]; then
  git clone https://github.com/openai/circuit_sparsity.git .external/circuit_sparsity
fi
git -C .external/circuit_sparsity checkout --detach dbf1fe0d27b76c19e10d2a715f28c2e5da535e08
"$PYTHON" -m sparse_circuit_repro audit
"$PYTHON" -m sparse_circuit_repro download-models
"$PYTHON" -m sparse_circuit_repro smoke
