#!/bin/bash
#SBATCH --job-name=field-cand-full
#SBATCH --partition=gold-batch
#SBATCH --array=0-21
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/field_candidate_full_%A_%a.out
#SBATCH --error=logs/field_candidate_full_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs inference_outputs/centered_field_acceleration_candidates .mplconfig

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

CANDIDATES_PATH="${CANDIDATES_PATH:-cluster/centered_acceleration_full_candidates.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-inference_outputs/centered_field_acceleration_candidates}"
DATA_PATH="${DATA_PATH:-data1055.npz}"
COUNT_STEPS="${COUNT_STEPS:-2998}"
START_FRAME="${START_FRAME:-0}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-250}"
DEVICE="${DEVICE:-auto}"
NCELLS="${NCELLS:-10}"
KCOUNT="${KCOUNT:-10}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OVERWRITE="${OVERWRITE:-0}"

mapfile -t CANDIDATE_LINES < <(grep -vE '^[[:space:]]*(#|$)' "${CANDIDATES_PATH}")
CANDIDATE_COUNT="${#CANDIDATE_LINES[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= CANDIDATE_COUNT )); then
    echo "TASK_ID=${TASK_ID} is outside candidate count ${CANDIDATE_COUNT}" >&2
    exit 2
fi

LINE="${CANDIDATE_LINES[$TASK_ID]}"
IFS=$'\t' read -r LABEL WEIGHTS MODEL_PATHS NOTES <<< "${LINE}"
if [[ -z "${LABEL}" || -z "${MODEL_PATHS}" ]]; then
    echo "Bad candidate line: ${LINE}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_PATH="${OUTPUT_ROOT}/${LABEL}_prediction.npz"
PLOT_PATH="${OUTPUT_ROOT}/${LABEL}_sqw_ncells${NCELLS}_kcount${KCOUNT}.png"
METRICS_PATH="${OUTPUT_ROOT}/${LABEL}_sqw_ncells${NCELLS}_kcount${KCOUNT}.txt"

MODEL_ARGS=()
IFS=';' read -r -a MODEL_PATH_ARRAY <<< "${MODEL_PATHS}"
for MODEL_PATH in "${MODEL_PATH_ARRAY[@]}"; do
    MODEL_ARGS+=(--model-path "${MODEL_PATH}")
done

if [[ "${WEIGHTS}" != "equal" ]]; then
    IFS=',' read -r -a MODEL_WEIGHT_ARRAY <<< "${WEIGHTS}"
    for MODEL_WEIGHT in "${MODEL_WEIGHT_ARRAY[@]}"; do
        MODEL_ARGS+=(--model-weight "${MODEL_WEIGHT}")
    done
fi

echo "TASK_ID=${TASK_ID}"
echo "LABEL=${LABEL}"
echo "WEIGHTS=${WEIGHTS}"
echo "MODEL_COUNT=${#MODEL_PATH_ARRAY[@]}"
echo "NOTES=${NOTES:-}"
printf 'MODEL_PATH=%s\n' "${MODEL_PATH_ARRAY[@]}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "PLOT_PATH=${PLOT_PATH}"
echo "COUNT_STEPS=${COUNT_STEPS}"

if [[ -f "${OUTPUT_PATH}" && "${OVERWRITE}" != "1" && "${SKIP_EXISTING}" == "1" ]]; then
    echo "SKIP inference because ${OUTPUT_PATH} already exists"
else
    python infer_field_rnn_centered_acceleration_ensemble.py \
        "${MODEL_ARGS[@]}" \
        --data-path "${DATA_PATH}" \
        --output-path "${OUTPUT_PATH}" \
        --count-steps "${COUNT_STEPS}" \
        --start-frame "${START_FRAME}" \
        --patch-shape ${PATCH_SHAPE:-3 3 3} \
        --patch-batch-size "${PATCH_BATCH_SIZE}" \
        --device "${DEVICE}" \
        --reference-output \
        --save-positions
fi

if [[ -f "${PLOT_PATH}" && "${OVERWRITE}" != "1" && "${SKIP_EXISTING}" == "1" ]]; then
    echo "SKIP plot because ${PLOT_PATH} already exists"
else
    python plot_sqw_comparison.py \
        --input-path "${OUTPUT_PATH}" \
        --output-path "${PLOT_PATH}" \
        --ncells "${NCELLS}" \
        --kcount "${KCOUNT}" | tee "${METRICS_PATH}"
fi
