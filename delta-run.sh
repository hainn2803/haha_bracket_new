#!/usr/bin/env bash
set -e
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"


conda activate /anvil/scratch/x-hnguyen23/env/openai_sparse

conda activate openai_sparse
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install -e .

python -u -m experiments.openai_sparse_plot.run_automatic_gradual_discovery_v12 --out-dir outputs/run_automatic_gradual_discovery_v12 --cuda 2>&1 | tee outputs/run_automatic_gradual_discovery_v12/terminal_output.txt

python -u -m experiments.openai_sparse_plot.run_find_downstream_r_late --out-dir outputs/run_find_downstream_r_late --cuda 2>&1 | tee outputs/run_find_downstream_r_late/terminal_output.txt

python -u -m experiments.openai_sparse_plot.run_bypass_restoration_test --out-dir outputs/run_bypass_restoration_test --cuda 2>&1 | tee outputs/run_bypass_restoration_test/terminal_output.txt

conda activate openai_sparse
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install -e .
python -u -m experiments.openai_sparse_plot.run_automatic_gradual_discovery_v16 --out-dir outputs/run_automatic_gradual_discovery_v16 --cuda 2>&1 | tee outputs/run_automatic_gradual_discovery_v16/terminal_output.txt

python -u -m experiments.openai_sparse_plot.run_automatic_gradual_discovery_v16_quote --out-dir outputs/run_automatic_gradual_discovery_v16_quote --cuda 2>&1 | tee outputs/run_automatic_gradual_discovery_v16_quote/terminal_output.txt

conda activate openai_sparse
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install -e .
python -u -m experiments.openai_sparse_plot.run_automatic_gradual_discovery_v18 --out-dir outputs/run_automatic_gradual_discovery_v18 --cuda 2>&1 | tee outputs/run_automatic_gradual_discovery_v18/terminal_output.txt

python -u -m experiments.openai_sparse_plot.run_automatic_gradual_discovery_v17_quote --out-dir outputs/run_automatic_gradual_discovery_v17_quote --cuda 2>&1 | tee outputs/run_automatic_gradual_discovery_v17_quote/terminal_output.txt