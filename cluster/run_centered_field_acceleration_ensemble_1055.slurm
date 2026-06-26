#!/bin/bash
#SBATCH --job-name=field-ens-accel
#SBATCH --partition=gold-batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/field_ens_accel_%j.out
#SBATCH --error=logs/field_ens_accel_%j.err

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

DEFAULT_MODEL_PATHS=(
    "models333_field_rnn_accnorm_sweep/rnn_h64_ec32_c2_d10k_ep100_lr1e-3/mean_norm_1.0176719132007475_rnn_field_rnn_acceleration_ec32_rh64_rl1_bidir_accnormchannel_cl2_k3.pth"
    "models333_field_rnn_accnorm_cyshift_center/mean_norm_0.7959334598258127_rnn_field_rnn_acceleration_ec32_rh64_rl1_bidir_accnormchannel_losscenter_cell_cyshift_cl2_k3.pth"
    "models333_field_rnn_accnorm_forcebalance_0p05_cyshift_center/mean_norm_0.6127292488948908_rnn_field_rnn_acceleration_ec32_rh64_rl1_bidir_accnormchannel_losscenter_cell_cyshift_fb0.05_cl2_k3.pth"
)

MODEL_ARGS=()
if [[ -n "${MODEL_PATHS:-}" ]]; then
    read -r -a MODEL_PATH_ARRAY <<< "${MODEL_PATHS}"
else
    MODEL_PATH_ARRAY=("${DEFAULT_MODEL_PATHS[@]}")
fi
for MODEL_PATH in "${MODEL_PATH_ARRAY[@]}"; do
    MODEL_ARGS+=(--model-path "${MODEL_PATH}")
done

if [[ -n "${MODEL_WEIGHTS:-}" ]]; then
    read -r -a MODEL_WEIGHT_ARRAY <<< "${MODEL_WEIGHTS}"
    for MODEL_WEIGHT in "${MODEL_WEIGHT_ARRAY[@]}"; do
        MODEL_ARGS+=(--model-weight "${MODEL_WEIGHT}")
    done
fi

DATA_PATH="${DATA_PATH:-data1055.npz}"
OUTPUT_PATH="${OUTPUT_PATH:-inference_outputs/centered_field_acceleration/data1055_centered_acceleration_ensemble_prediction.npz}"

python infer_field_rnn_centered_acceleration_ensemble.py \
    "${MODEL_ARGS[@]}" \
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
