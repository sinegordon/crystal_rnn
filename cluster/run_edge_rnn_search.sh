#!/bin/bash
set -euo pipefail

MODE="${1:-submit}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(dirname "$0")/.."
fi

mkdir -p logs

RUN_LABEL="${RUN_LABEL:-edge_rnn_333_shell2_10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-inference_outputs/${RUN_LABEL}}"
MODELS_DIR="${MODELS_DIR:-models333_${RUN_LABEL}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
TOP10_PATH="${TOP10_PATH:-${OUTPUT_ROOT}/top10.txt}"
MODEL_COUNT="${MODEL_COUNT:-10}"
BASE_SEED="${BASE_SEED:-20260901}"

if [[ "${MODE}" == "submit" ]]; then
    TRAIN_NODELIST_ARGS=()
    COLLECT_NODELIST_ARGS=()
    if [[ -n "${TRAIN_NODELIST:-node54.cluster}" ]]; then
        TRAIN_NODELIST_ARGS=(--nodelist="${TRAIN_NODELIST:-node54.cluster}")
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
    echo "DATA_PATH=${DATA_PATH:-data333.npz}"
    echo "EVAL_DATA_PATH=${EVAL_DATA_PATH:-}"
    echo "ARCHITECTURE=${ARCHITECTURE:-edge}"
    echo "TRAINING_TARGET=${TRAINING_TARGET:-displacement}"
    echo "RNN_READOUT_MODE=${RNN_READOUT_MODE:-last-output}"
    echo "TEMPORAL_ARCHITECTURE=${TEMPORAL_ARCHITECTURE:-stacked}"
    echo "TEMPORAL_INPUT_MODE=${TEMPORAL_INPUT_MODE:-absolute-pair}"
    echo "DISPLACEMENT_MOMENT_LOSS_WEIGHT=${DISPLACEMENT_MOMENT_LOSS_WEIGHT:-0.0}"
    echo "DISPLACEMENT_MOMENT_MEAN_WEIGHT=${DISPLACEMENT_MOMENT_MEAN_WEIGHT:-1.0}"
    echo "DISPLACEMENT_MOMENT_STD_WEIGHT=${DISPLACEMENT_MOMENT_STD_WEIGHT:-1.0}"
    echo "DISPLACEMENT_MOMENT_RMS_WEIGHT=${DISPLACEMENT_MOMENT_RMS_WEIGHT:-0.0}"
    echo "DISPLACEMENT_MOMENT_COMPONENT_WEIGHTS=${DISPLACEMENT_MOMENT_COMPONENT_WEIGHTS:-1 1 1}"
    echo "POWER_MEAN_LOSS_WEIGHT=${POWER_MEAN_LOSS_WEIGHT:-0.0}"
    echo "Q_POWER_LOSS_WEIGHT=${Q_POWER_LOSS_WEIGHT:-0.0}"
    echo "Q_POWER_LOSS_MODE=${Q_POWER_LOSS_MODE:-positive-excess}"
    echo "Q_POWER_LOSS_SAMPLE_COUNT=${Q_POWER_LOSS_SAMPLE_COUNT:-2}"
    echo "Q_POWER_LOSS_INTERVAL=${Q_POWER_LOSS_INTERVAL:-10}"
    echo "Q_POWER_LOSS_MARGIN=${Q_POWER_LOSS_MARGIN:-0.0}"
    echo "Q_POWER_LOSS_EXCLUDE_Q_ZERO=${Q_POWER_LOSS_EXCLUDE_Q_ZERO:-true}"
    echo "ACCELERATION_RMS_LOSS_WEIGHT=${ACCELERATION_RMS_LOSS_WEIGHT:-0.0}"
    echo "ACCELERATION_BATCH_RMS_LOSS_WEIGHT=${ACCELERATION_BATCH_RMS_LOSS_WEIGHT:-0.0}"
    echo "ACCELERATION_TAIL_LOSS_WEIGHT=${ACCELERATION_TAIL_LOSS_WEIGHT:-0.0}"
    echo "ACCELERATION_OVER_RMS_LOSS_WEIGHT=${ACCELERATION_OVER_RMS_LOSS_WEIGHT:-0.0}"
    echo "ACCELERATION_UNDER_RMS_LOSS_WEIGHT=${ACCELERATION_UNDER_RMS_LOSS_WEIGHT:-0.0}"
    echo "ACCELERATION_OVER_RMS_LOSS_POWER=${ACCELERATION_OVER_RMS_LOSS_POWER:-2.0}"
    echo "ACCELERATION_UNDER_RMS_LOSS_POWER=${ACCELERATION_UNDER_RMS_LOSS_POWER:-2.0}"
    echo "CURL_LOSS_WEIGHT=${CURL_LOSS_WEIGHT:-0.0}"
    echo "CURL_LOSS_SAMPLE_COUNT=${CURL_LOSS_SAMPLE_COUNT:-4}"
    echo "CURL_LOSS_INTERVAL=${CURL_LOSS_INTERVAL:-1}"
    echo "REFERENCE_PRESSURE_LOSS_WEIGHT=${REFERENCE_PRESSURE_LOSS_WEIGHT:-0.0}"
    echo "REFERENCE_PRESSURE_TARGET=${REFERENCE_PRESSURE_TARGET:-0.0}"
    echo "REFERENCE_PRESSURE_LOSS_SCALE=${REFERENCE_PRESSURE_LOSS_SCALE:-1.0}"
    echo "ACCELERATION_SCORE_WEIGHT=${ACCELERATION_SCORE_WEIGHT:-0.0}"
    echo "DEVICE=${DEVICE:-cuda}"

    EVAL_ARGS=()
    if [[ -n "${EVAL_DATA_PATH:-}" ]]; then
        EVAL_ARGS=(--eval-data-path "${EVAL_DATA_PATH}")
    fi
    Q_POWER_EXCLUDE_ARGS=(--q-power-loss-exclude-q-zero)
    if [[ "${Q_POWER_LOSS_EXCLUDE_Q_ZERO:-true}" == "false" || "${Q_POWER_LOSS_EXCLUDE_Q_ZERO:-true}" == "0" ]]; then
        Q_POWER_EXCLUDE_ARGS=(--no-q-power-loss-exclude-q-zero)
    fi

    python find_edge_rnn_models.py 1 "${RNN_TYPE:-RNN}" \
        --data-path "${DATA_PATH:-data333.npz}" \
        "${EVAL_ARGS[@]}" \
        --models-dir "${MODELS_DIR}" \
        --metrics-path "${METRICS_PATH}" \
        --count-steps "${COUNT_STEPS:-2000}" \
        --count-run "${COUNT_RUN:-3}" \
        --delta-frames "${DELTA_FRAMES:-30000}" \
        --epochs "${EPOCHS:-100}" \
        --batch-size "${BATCH_SIZE:-256}" \
        --data-len "${DATA_LEN:-1.0}" \
        --learning-rate "${LEARNING_RATE:-0.001}" \
        --hidden-size "${HIDDEN_SIZE:-128}" \
        --rnn-layers "${RNN_LAYERS:-1}" \
        --rnn-readout-mode "${RNN_READOUT_MODE:-last-output}" \
        --temporal-architecture "${TEMPORAL_ARCHITECTURE:-stacked}" \
        --temporal-input-mode "${TEMPORAL_INPUT_MODE:-absolute-pair}" \
        --architecture "${ARCHITECTURE:-edge}" \
        --neighbor-shells "${NEIGHBOR_SHELLS:-2}" \
        --cutoff-scale "${CUTOFF_SCALE:-1.05}" \
        --acceleration-normalization "${ACCELERATION_NORMALIZATION:-channel}" \
        --training-target "${TRAINING_TARGET:-displacement}" \
        --velocity-score-weight "${VELOCITY_SCORE_WEIGHT:-0.0}" \
        --acceleration-score-weight "${ACCELERATION_SCORE_WEIGHT:-0.0}" \
        --displacement-moment-loss-weight "${DISPLACEMENT_MOMENT_LOSS_WEIGHT:-0.0}" \
        --displacement-moment-mean-weight "${DISPLACEMENT_MOMENT_MEAN_WEIGHT:-1.0}" \
        --displacement-moment-std-weight "${DISPLACEMENT_MOMENT_STD_WEIGHT:-1.0}" \
        --displacement-moment-rms-weight "${DISPLACEMENT_MOMENT_RMS_WEIGHT:-0.0}" \
        --displacement-moment-component-weights ${DISPLACEMENT_MOMENT_COMPONENT_WEIGHTS:-1 1 1} \
        --displacement-moment-loss-epsilon "${DISPLACEMENT_MOMENT_LOSS_EPSILON:-1e-12}" \
        --power-mean-loss-weight "${POWER_MEAN_LOSS_WEIGHT:-0.0}" \
        --power-mean-loss-epsilon "${POWER_MEAN_LOSS_EPSILON:-1e-12}" \
        --q-power-loss-weight "${Q_POWER_LOSS_WEIGHT:-0.0}" \
        --q-power-loss-mode "${Q_POWER_LOSS_MODE:-positive-excess}" \
        --q-power-loss-sample-count "${Q_POWER_LOSS_SAMPLE_COUNT:-2}" \
        --q-power-loss-interval "${Q_POWER_LOSS_INTERVAL:-10}" \
        --q-power-loss-margin "${Q_POWER_LOSS_MARGIN:-0.0}" \
        --q-power-loss-epsilon "${Q_POWER_LOSS_EPSILON:-1e-12}" \
        "${Q_POWER_EXCLUDE_ARGS[@]}" \
        --acceleration-rms-loss-weight "${ACCELERATION_RMS_LOSS_WEIGHT:-0.0}" \
        --acceleration-batch-rms-loss-weight "${ACCELERATION_BATCH_RMS_LOSS_WEIGHT:-0.0}" \
        --acceleration-tail-loss-weight "${ACCELERATION_TAIL_LOSS_WEIGHT:-0.0}" \
        --acceleration-over-rms-loss-weight "${ACCELERATION_OVER_RMS_LOSS_WEIGHT:-0.0}" \
        --acceleration-under-rms-loss-weight "${ACCELERATION_UNDER_RMS_LOSS_WEIGHT:-0.0}" \
        --acceleration-over-rms-loss-power "${ACCELERATION_OVER_RMS_LOSS_POWER:-2.0}" \
        --acceleration-under-rms-loss-power "${ACCELERATION_UNDER_RMS_LOSS_POWER:-2.0}" \
        --rms-loss-epsilon "${RMS_LOSS_EPSILON:-1e-12}" \
        --curl-loss-weight "${CURL_LOSS_WEIGHT:-0.0}" \
        --curl-loss-sample-count "${CURL_LOSS_SAMPLE_COUNT:-4}" \
        --curl-loss-interval "${CURL_LOSS_INTERVAL:-1}" \
        --curl-loss-epsilon "${CURL_LOSS_EPSILON:-1e-12}" \
        --reference-pressure-loss-weight "${REFERENCE_PRESSURE_LOSS_WEIGHT:-0.0}" \
        --reference-pressure-target "${REFERENCE_PRESSURE_TARGET:-0.0}" \
        --reference-pressure-loss-scale "${REFERENCE_PRESSURE_LOSS_SCALE:-1.0}" \
        --velocity-window-frames "${VELOCITY_WINDOW_FRAMES:-10}" \
        --velocity-hist-bins "${VELOCITY_HIST_BINS:-80}" \
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
