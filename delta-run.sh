#!/usr/bin/env bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate openai_sparse
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install -e .

python -u -m experiments.openai_sparse_plot.run_automatic_gradual_discovery_v10 --out-dir outputs/run_automatic_gradual_discovery_v10 --cuda 2>&1 | tee outputs/run_automatic_gradual_discovery_v10/terminal_output.txt