#!/bin/bash
#SBATCH --job-name=crystal-accnorm
#SBATCH --partition=gold-batch
#SBATCH --array=0-19
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/field_accnorm_%A_%a.out
#SBATCH --error=logs/field_accnorm_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# The cluster GPU node is expected to be node54 when using the gold-batch queue.

if [[ -n "${MODULE_LOAD:-}" ]]; then
    # Example: MODULE_LOAD="cuda/12.1 anaconda3" sbatch ...
    module load ${MODULE_LOAD}
fi

if [[ -n "${CONDA_SH:-}" ]]; then
    # Example: CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh" sbatch ...
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
BASE_SEED="${BASE_SEED:-20260518}"
SEED=$((BASE_SEED + TASK_ID))

OUTPUT_ROOT="${OUTPUT_ROOT:-inference_outputs/field_rnn_accnorm_cluster}"
MODELS_DIR="${MODELS_DIR:-models333_field_rnn_accnorm_cluster}"
mkdir -p "${OUTPUT_ROOT}" "${MODELS_DIR}"

CYCLIC_SHIFT_ARGS=(--no-cyclic-shift-augmentation)
case "${CYCLIC_SHIFT_AUGMENTATION:-0}" in
    1|true|TRUE|yes|YES|on|ON)
        CYCLIC_SHIFT_ARGS=(--cyclic-shift-augmentation)
        ;;
esac

python find_field_rnn_models.py 1 RNN \
    --data-path "${DATA_PATH:-data333.npz}" \
    --models-dir "${MODELS_DIR}" \
    --metrics-path "${OUTPUT_ROOT}/metrics_${TASK_ID}.tsv" \
    --count-steps "${COUNT_STEPS:-2000}" \
    --count-run "${COUNT_RUN:-1}" \
    --velocity-score-weight "${VELOCITY_SCORE_WEIGHT:-0.0}" \
    --velocity-window-frames "${VELOCITY_WINDOW_FRAMES:-10}" \
    --velocity-hist-bins "${VELOCITY_HIST_BINS:-80}" \
    --velocity-max-end-speed-ratio "${VELOCITY_MAX_END_SPEED_RATIO:-inf}" \
    --delta-frames "${DELTA_FRAMES:-10000}" \
    --encoder-channels "${ENCODER_CHANNELS:-32}" \
    --rnn-hidden-size "${RNN_HIDDEN_SIZE:-64}" \
    --rnn-layers "${RNN_LAYERS:-1}" \
    --conv-layers "${CONV_LAYERS:-1}" \
    --epochs "${EPOCHS:-30}" \
    --batch-size "${BATCH_SIZE:-128}" \
    --data-len "${DATA_LEN:-0.2}" \
    --learning-rate "${LEARNING_RATE:-0.001}" \
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
    --plot-output-dir "${OUTPUT_ROOT}/plots_${TASK_ID}" \
    --seed "${SEED}" \
    "$@"
