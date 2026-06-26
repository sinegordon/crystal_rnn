#!/bin/bash
#SBATCH --job-name=oldbest-thermo1055
#SBATCH --partition=gold-batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/oldbest_thermo1055_%j.out
#SBATCH --error=logs/oldbest_thermo1055_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

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
mkdir -p "${MPLCONFIGDIR}"

MODEL_PATH="${MODEL_PATH:-models333_field_rnn_accnorm_cl2_d30k_ep100/mean_norm_0.9217232867679755_rnn_field_rnn_acceleration_ec32_rh64_rl1_bidir_accnormchannel_cl2_k3.pth}"
DATA_PATH="${DATA_PATH:-data1055.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-inference_outputs/old_best_1055_thermostat}"
PREDICTION_PATH="${OUTPUT_ROOT}/old_best_1055_thermostat_prediction.npz"
PLOT_PATH="${OUTPUT_ROOT}/old_best_1055_thermostat_sqw.png"
METRICS_PATH="${OUTPUT_ROOT}/old_best_1055_thermostat_sqw.txt"
START_FRAME="${START_FRAME:-0}"
COUNT_STEPS="${COUNT_STEPS:-2998}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-250}"
THERMOSTAT_INTERVAL="${THERMOSTAT_INTERVAL:-1}"
THERMOSTAT_COUPLING="${THERMOSTAT_COUPLING:-0.1}"
NCELLS="${NCELLS:-10}"
KCOUNT="${KCOUNT:-10}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_ROOT}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_PATH=${DATA_PATH}"
echo "PREDICTION_PATH=${PREDICTION_PATH}"
echo "PLOT_PATH=${PLOT_PATH}"
echo "COUNT_STEPS=${COUNT_STEPS}"
echo "THERMOSTAT_INTERVAL=${THERMOSTAT_INTERVAL}"
echo "THERMOSTAT_COUPLING=${THERMOSTAT_COUPLING}"

python infer_field_rnn_centered_acceleration.py \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --output-path "${PREDICTION_PATH}" \
    --count-steps "${COUNT_STEPS}" \
    --start-frame "${START_FRAME}" \
    --patch-shape 3 3 3 \
    --patch-batch-size "${PATCH_BATCH_SIZE}" \
    --velocity-thermostat-interval "${THERMOSTAT_INTERVAL}" \
    --velocity-thermostat-coupling "${THERMOSTAT_COUPLING}" \
    --device "${DEVICE}" \
    --reference-output \
    --save-positions

python plot_sqw_comparison.py \
    --input-path "${PREDICTION_PATH}" \
    --output-path "${PLOT_PATH}" \
    --ncells "${NCELLS}" \
    --kcount "${KCOUNT}" | tee "${METRICS_PATH}"

echo "PREDICTION_PATH=${PREDICTION_PATH}"
echo "PLOT_PATH=${PLOT_PATH}"
echo "METRICS_PATH=${METRICS_PATH}"
