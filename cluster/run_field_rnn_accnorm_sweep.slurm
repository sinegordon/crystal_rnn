#!/bin/bash
#SBATCH --job-name=crystal-accsweep
#SBATCH --partition=gold-batch
#SBATCH --array=0-23
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/field_accsweep_%A_%a.out
#SBATCH --error=logs/field_accsweep_%A_%a.err

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

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEEDS_PER_CONFIG="${SEEDS_PER_CONFIG:-3}"
CONFIG_ID=$((TASK_ID / SEEDS_PER_CONFIG))
SEED_OFFSET=$((TASK_ID % SEEDS_PER_CONFIG))
BASE_SEED="${BASE_SEED:-20260601}"
SEED=$((BASE_SEED + TASK_ID))

RNN_TYPE="RNN"
ENCODER_CHANNELS=32
RNN_HIDDEN_SIZE=64
RNN_LAYERS=1
CONV_LAYERS=1
DELTA_FRAMES=10000
EPOCHS=100
BATCH_SIZE=256
DATA_LEN=1.0
LEARNING_RATE=0.001
ACTIVATION="elu"

case "${CONFIG_ID}" in
    0)
        LABEL="rnn_h64_ec32_c1_d10k_ep100_lr1e-3"
        ;;
    1)
        LABEL="rnn_h128_ec32_c1_d10k_ep100_lr1e-3"
        RNN_HIDDEN_SIZE=128
        ;;
    2)
        LABEL="gru_h64_ec32_c1_d10k_ep100_lr1e-3"
        RNN_TYPE="GRU"
        ;;
    3)
        LABEL="gru_h128_ec32_c1_d10k_ep100_lr1e-3"
        RNN_TYPE="GRU"
        RNN_HIDDEN_SIZE=128
        ;;
    4)
        LABEL="rnn_h64_ec64_c1_d10k_ep100_lr1e-3"
        ENCODER_CHANNELS=64
        ;;
    5)
        LABEL="rnn_h64_ec32_c2_d10k_ep100_lr1e-3"
        CONV_LAYERS=2
        ;;
    6)
        LABEL="rnn_h64_ec32_c1_d30k_ep80_lr1e-3"
        DELTA_FRAMES=30000
        EPOCHS=80
        ;;
    7)
        LABEL="rnn_h64_ec32_c1_d10k_ep150_lr3e-4"
        EPOCHS=150
        LEARNING_RATE=0.0003
        ;;
    *)
        echo "Unsupported CONFIG_ID=${CONFIG_ID}; adjust --array or SEEDS_PER_CONFIG." >&2
        exit 2
        ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-inference_outputs/field_rnn_accnorm_sweep}"
MODELS_ROOT="${MODELS_ROOT:-models333_field_rnn_accnorm_sweep}"
MODELS_DIR="${MODELS_ROOT}/${LABEL}"
PLOT_DIR="${OUTPUT_ROOT}/${LABEL}/plots_${SEED_OFFSET}"
METRICS_PATH="${OUTPUT_ROOT}/${LABEL}/metrics_${SEED_OFFSET}.tsv"
mkdir -p "${OUTPUT_ROOT}/${LABEL}" "${MODELS_DIR}" "${PLOT_DIR}"

echo "TASK_ID=${TASK_ID}"
echo "CONFIG_ID=${CONFIG_ID}"
echo "LABEL=${LABEL}"
echo "SEED=${SEED}"
echo "RNN_TYPE=${RNN_TYPE}"
echo "ENCODER_CHANNELS=${ENCODER_CHANNELS}"
echo "RNN_HIDDEN_SIZE=${RNN_HIDDEN_SIZE}"
echo "RNN_LAYERS=${RNN_LAYERS}"
echo "CONV_LAYERS=${CONV_LAYERS}"
echo "DELTA_FRAMES=${DELTA_FRAMES}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "DATA_LEN=${DATA_LEN}"
echo "LEARNING_RATE=${LEARNING_RATE}"

CYCLIC_SHIFT_ARGS=(--no-cyclic-shift-augmentation)
case "${CYCLIC_SHIFT_AUGMENTATION:-0}" in
    1|true|TRUE|yes|YES|on|ON)
        CYCLIC_SHIFT_ARGS=(--cyclic-shift-augmentation)
        ;;
esac

python find_field_rnn_models.py 1 "${RNN_TYPE}" \
    --data-path "${DATA_PATH:-data333.npz}" \
    --models-dir "${MODELS_DIR}" \
    --metrics-path "${METRICS_PATH}" \
    --count-steps "${COUNT_STEPS:-2000}" \
    --count-run "${COUNT_RUN:-1}" \
    --velocity-score-weight "${VELOCITY_SCORE_WEIGHT:-0.0}" \
    --velocity-window-frames "${VELOCITY_WINDOW_FRAMES:-10}" \
    --velocity-hist-bins "${VELOCITY_HIST_BINS:-80}" \
    --velocity-max-end-speed-ratio "${VELOCITY_MAX_END_SPEED_RATIO:-inf}" \
    --delta-frames "${DELTA_FRAMES}" \
    --encoder-channels "${ENCODER_CHANNELS}" \
    --rnn-hidden-size "${RNN_HIDDEN_SIZE}" \
    --rnn-layers "${RNN_LAYERS}" \
    --conv-layers "${CONV_LAYERS}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --data-len "${DATA_LEN}" \
    --learning-rate "${LEARNING_RATE}" \
    --activation "${ACTIVATION}" \
    --target-mode acceleration \
    --acceleration-normalization "${ACCELERATION_NORMALIZATION:-channel}" \
    --loss-region "${LOSS_REGION:-all}" \
    "${CYCLIC_SHIFT_ARGS[@]}" \
    --input-transform "${INPUT_TRANSFORM:-absolute}" \
    --input-relative-scale "${INPUT_RELATIVE_SCALE:-1.0}" \
    --force-balance-loss-weight "${FORCE_BALANCE_LOSS_WEIGHT:-0.0}" \
    --acceleration-rms-loss-weight "${ACCELERATION_RMS_LOSS_WEIGHT:-0.0}" \
    --velocity-rms-loss-weight "${VELOCITY_RMS_LOSS_WEIGHT:-0.0}" \
    --rms-loss-epsilon "${RMS_LOSS_EPSILON:-1e-12}" \
    --device "${DEVICE:-auto}" \
    --bidirectional \
    --save-all \
    --plot-all \
    --plot-output-dir "${PLOT_DIR}" \
    --seed "${SEED}" \
    "$@"
