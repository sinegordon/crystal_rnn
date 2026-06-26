#!/bin/bash
#SBATCH --job-name=cl2-d30k-1055
#SBATCH --partition=gold-batch
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cl2_d30k_1055_%j.out
#SBATCH --error=logs/cl2_d30k_1055_%j.err

set -euo pipefail

# Reuse the already tested train -> 1055 inference -> S(q,w) montage workflow,
# but keep the d30k outputs and Slurm logs separate from the d60k experiment.
export RUN_LABEL="${RUN_LABEL:-cl2_d30k_10x_full1055}"
export DELTA_FRAMES="${DELTA_FRAMES:-30000}"
export BASE_SEED="${BASE_SEED:-20260720}"

exec bash cluster/run_cl2_d60k_train_infer_1055_sweep.slurm "$@"
