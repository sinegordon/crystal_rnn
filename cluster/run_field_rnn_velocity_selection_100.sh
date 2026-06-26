#!/bin/bash
set -euo pipefail

MODE="${1:-submit}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(dirname "$0")/.."
fi

mkdir -p logs

RUN_LABEL="${RUN_LABEL:-field_rnn_cl2_d30k_velocity_w1_100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-inference_outputs/${RUN_LABEL}}"
MODELS_DIR="${MODELS_DIR:-models333_${RUN_LABEL}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
TOP10_PATH="${TOP10_PATH:-${OUTPUT_ROOT}/top10.txt}"
MODEL_COUNT="${MODEL_COUNT:-100}"
BASE_SEED="${BASE_SEED:-20260801}"

if [[ "${MODE}" == "submit" ]]; then
    TRAIN_NODELIST_ARGS=()
    COLLECT_NODELIST_ARGS=()
    if [[ -n "${TRAIN_NODELIST:-}" ]]; then
        TRAIN_NODELIST_ARGS=(--nodelist="${TRAIN_NODELIST}")
    fi
    if [[ -n "${COLLECT_NODELIST:-}" ]]; then
        COLLECT_NODELIST_ARGS=(--nodelist="${COLLECT_NODELIST}")
    fi

    ARRAY_JOB_ID="$(
        sbatch \
            --parsable \
            --job-name="${RUN_LABEL}" \
            --partition="${TRAIN_PARTITION:-gold-batch}" \
            "${TRAIN_NODELIST_ARGS[@]}" \
            --array="0-$((MODEL_COUNT - 1))" \
            --time="${TRAIN_TIME:-06:00:00}" \
            --cpus-per-task="${CPUS_PER_TASK:-4}" \
            --mem="${TRAIN_MEM:-16G}" \
            --gres="${TRAIN_GRES:-gpu:1}" \
            --output="logs/${RUN_LABEL}_%A_%a.out" \
            --error="logs/${RUN_LABEL}_%A_%a.err" \
            --export=ALL \
            "$0" worker
    )"
    COLLECT_JOB_ID="$(
        sbatch \
            --parsable \
            --job-name="${RUN_LABEL}_collect" \
            --partition="${COLLECT_PARTITION:-gold-batch}" \
            "${COLLECT_NODELIST_ARGS[@]}" \
            --dependency="afterany:${ARRAY_JOB_ID}" \
            --time="${COLLECT_TIME:-00:20:00}" \
            --cpus-per-task=1 \
            --mem="${COLLECT_MEM:-4G}" \
            --output="logs/${RUN_LABEL}_collect_%j.out" \
            --error="logs/${RUN_LABEL}_collect_%j.err" \
            --export=ALL \
            "$0" collect
    )"
    echo "ARRAY_JOB_ID=${ARRAY_JOB_ID}"
    echo "COLLECT_JOB_ID=${COLLECT_JOB_ID}"
    echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
    echo "MODELS_DIR=${MODELS_DIR}"
    echo "SUMMARY_PATH=${SUMMARY_PATH}"
    echo "TOP10_PATH=${TOP10_PATH}"
    exit 0
fi

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
mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_ROOT}" "${MODELS_DIR}"

if [[ "${MODE}" == "worker" ]]; then
    TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    SEED=$((BASE_SEED + TASK_ID))
    PLOT_DIR="${OUTPUT_ROOT}/plots_${TASK_ID}"
    METRICS_PATH="${OUTPUT_ROOT}/metrics_${TASK_ID}.tsv"
    mkdir -p "${PLOT_DIR}"

    echo "TASK_ID=${TASK_ID}"
    echo "SEED=${SEED}"
    echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
    echo "MODELS_DIR=${MODELS_DIR}"
    echo "METRICS_PATH=${METRICS_PATH}"

    python find_field_rnn_models.py 1 RNN \
        --data-path "${DATA_PATH:-data333.npz}" \
        --models-dir "${MODELS_DIR}" \
        --metrics-path "${METRICS_PATH}" \
        --count-steps "${COUNT_STEPS:-2000}" \
        --count-run "${COUNT_RUN:-1}" \
        --velocity-score-weight "${VELOCITY_SCORE_WEIGHT:-1.0}" \
        --velocity-window-frames "${VELOCITY_WINDOW_FRAMES:-10}" \
        --velocity-hist-bins "${VELOCITY_HIST_BINS:-80}" \
        --velocity-max-end-speed-ratio "${VELOCITY_MAX_END_SPEED_RATIO:-inf}" \
        --delta-frames "${DELTA_FRAMES:-30000}" \
        --encoder-channels "${ENCODER_CHANNELS:-32}" \
        --rnn-hidden-size "${RNN_HIDDEN_SIZE:-64}" \
        --rnn-layers "${RNN_LAYERS:-1}" \
        --conv-layers "${CONV_LAYERS:-2}" \
        --epochs "${EPOCHS:-100}" \
        --batch-size "${BATCH_SIZE:-256}" \
        --data-len "${DATA_LEN:-1.0}" \
        --learning-rate "${LEARNING_RATE:-0.001}" \
        --target-mode acceleration \
        --acceleration-normalization "${ACCELERATION_NORMALIZATION:-channel}" \
        --loss-region "${LOSS_REGION:-all}" \
        --no-cyclic-shift-augmentation \
        --input-transform "${INPUT_TRANSFORM:-absolute}" \
        --input-relative-scale "${INPUT_RELATIVE_SCALE:-1.0}" \
        --force-balance-loss-weight "${FORCE_BALANCE_LOSS_WEIGHT:-0.0}" \
        --acceleration-rms-loss-weight "${ACCELERATION_RMS_LOSS_WEIGHT:-0.0}" \
        --velocity-rms-loss-weight "${VELOCITY_RMS_LOSS_WEIGHT:-0.0}" \
        --rms-loss-epsilon "${RMS_LOSS_EPSILON:-1e-12}" \
        --low-q-stiffness-loss-weight "${LOW_Q_STIFFNESS_LOSS_WEIGHT:-0.0}" \
        --low-q-stiffness-max-shell "${LOW_Q_STIFFNESS_MAX_SHELL:-1}" \
        --low-q-stiffness-epsilon "${LOW_Q_STIFFNESS_EPSILON:-1e-8}" \
        --device "${DEVICE:-cuda}" \
        --bidirectional \
        --save-all \
        --plot-all \
        --plot-output-dir "${PLOT_DIR}" \
        --seed "${SEED}"

    echo "DONE worker TASK_ID=${TASK_ID}"
    exit 0
fi

if [[ "${MODE}" == "collect" ]]; then
    echo "Collecting metrics from ${OUTPUT_ROOT}"
    python cluster/collect_field_rnn_metrics.py \
        --metrics-dir "${OUTPUT_ROOT}" \
        --output-path "${SUMMARY_PATH}" | tee "${TOP10_PATH}"
    echo "SUMMARY_PATH=${SUMMARY_PATH}"
    echo "TOP10_PATH=${TOP10_PATH}"
    exit 0
fi

echo "Unsupported mode: ${MODE}" >&2
exit 2
