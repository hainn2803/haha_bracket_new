#!/usr/bin/env bash
set -e
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"


conda activate /anvil/scratch/x-hnguyen23/env/openai_sparse
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install -e .

python -u -m experiments.openai_sparse_plot.run_automatic_gradual_discovery_v10 --out-dir outputs/run_automatic_gradual_discovery_v10 --cuda 2>&1 | tee outputs/run_automatic_gradual_discovery_v10/terminal_output.txt

python -u -m experiments.openai_sparse_plot.run_find_downstream_r_late --out-dir outputs/run_find_downstream_r_late --cuda 2>&1 | tee outputs/run_find_downstream_r_late/terminal_output.txt

python -u -m experiments.openai_sparse_plot.run_bypass_restoration_test --out-dir outputs/run_bypass_restoration_test --cuda 2>&1 | tee outputs/run_bypass_restoration_test/terminal_output.txt
