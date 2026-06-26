#!/bin/bash
#SBATCH --job-name=ref_eam_cons
#SBATCH --partition=gold-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ref_eam_conservativity_%j.out
#SBATCH --error=logs/ref_eam_conservativity_%j.err

source ~/.bashrc
set -euo pipefail
conda activate "${CONDA_ENV:-torch}"

cd "${WORKDIR:-$HOME/crystal_rnn_accnorm}"
mkdir -p logs

python diagnose_force_conservativity.py \
    --source eam-reference \
    --eam-backend ase \
    --eam-potential "${EAM_POTENTIAL:-Cu_u3.eam}" \
    --data-path "${DATA_PATH:-data333_force.npz}" \
    --output-dir "${OUTPUT_DIR:-inference_outputs/conservativity/reference_eam_eps005_jac100}" \
    --samples "${SAMPLES:-200}" \
    --loops-per-sample "${LOOPS_PER_SAMPLE:-20}" \
    --epsilon "${EPSILON:-0.05}" \
    --jacobian-samples "${JACOBIAN_SAMPLES:-100}" \
    --jacobian-epsilon "${JACOBIAN_EPSILON:-1e-4}" \
    --device cpu
