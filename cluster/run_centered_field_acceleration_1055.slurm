#!/bin/bash
#SBATCH --job-name=field-center-accel
#SBATCH --partition=gold-batch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/field_center_accel_%j.out
#SBATCH --error=logs/field_center_accel_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs inference_outputs/centered_field_acceleration .mplconfig

if [[ -n "${CONDA_SH:-}" ]]; then
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV:-torch}"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-torch}"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

MODEL_PATH="${MODEL_PATH:-models333_field_rnn_accnorm_sweep/rnn_h64_ec32_c2_d10k_ep100_lr1e-3/mean_norm_1.0176719132007475_rnn_field_rnn_acceleration_ec32_rh64_rl1_bidir_accnormchannel_cl2_k3.pth}"
DATA_PATH="${DATA_PATH:-data1055.npz}"
OUTPUT_PATH="${OUTPUT_PATH:-inference_outputs/centered_field_acceleration/data1055_cl2_best_centered_acceleration_prediction.npz}"

python infer_field_rnn_centered_acceleration.py \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --count-steps "${COUNT_STEPS:-2998}" \
    --start-frame "${START_FRAME:-0}" \
    --patch-shape ${PATCH_SHAPE:-3 3 3} \
    --patch-batch-size "${PATCH_BATCH_SIZE:-250}" \
    --device "${DEVICE:-auto}" \
    --reference-output \
    --save-positions \
    "$@"
