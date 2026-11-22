#!/usr/bin/env bash
# One-time setup on a remote machine (e.g. a Babbage remote-development session).
# From the repo root:  bash challenge-1-ml-decoder-kit/bench/setup_remote.sh
# Creates .venv at the repo root, installs Rust if needed (user-level, no admin), builds
# relay-bp, runs the tests and prints the machine's CPU and GPU resources.
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! command -v cargo >/dev/null 2>&1; then
  [ -f "$HOME/.cargo/env" ] || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  source "$HOME/.cargo/env"
fi

PY=${PYTHON:-python3}
"$PY" -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r challenge-1-ml-decoder-kit/requirements.txt pandas matplotlib scikit-learn pytest

cd challenge-1-ml-decoder-kit
python -m pytest -q bench/test_bench.py
python -c "import relay_bp, ldpc, stim; print('relay_bp, ldpc, stim import OK')"

echo "--- resources ---"
python -c "import os; print('python', __import__('sys').version.split()[0], '| CPUs usable:', len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count())"
lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)" || true
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no nvidia-smi"
