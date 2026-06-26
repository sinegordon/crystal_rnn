#!/bin/bash
#SBATCH --job-name=cl2-d60k-1055
#SBATCH --partition=gold-batch
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cl2_d60k_1055_%j.out
#SBATCH --error=logs/cl2_d60k_1055_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs .mplconfig

if [[ -n "${CONDA_SH:-}" ]]; then
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV:-torch}"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-torch}"
fi

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "${MPLCONFIGDIR}"

RUN_LABEL="${RUN_LABEL:-cl2_d60k_10x_full1055}"
OUTPUT_ROOT="${OUTPUT_ROOT:-inference_outputs/${RUN_LABEL}}"
MODELS_DIR="${MODELS_DIR:-models333_field_rnn_accnorm_${RUN_LABEL}}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-data333.npz}"
INFERENCE_DATA_PATH="${INFERENCE_DATA_PATH:-data1055.npz}"
MODEL_COUNT="${MODEL_COUNT:-10}"
BASE_SEED="${BASE_SEED:-20260620}"

DELTA_FRAMES="${DELTA_FRAMES:-60000}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DATA_LEN="${DATA_LEN:-1.0}"
COUNT_RUN="${COUNT_RUN:-1}"
TRAIN_COUNT_STEPS="${TRAIN_COUNT_STEPS:-2000}"
INFERENCE_COUNT_STEPS="${INFERENCE_COUNT_STEPS:-2998}"
START_FRAME="${START_FRAME:-0}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-250}"
NCELLS="${NCELLS:-10}"
KCOUNT="${KCOUNT:-10}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_ROOT}" "${MODELS_DIR}"
SUMMARY_PATH="${OUTPUT_ROOT}/summary.tsv"
MONTAGE_PATH="${OUTPUT_ROOT}/sqw_1055_montage.png"

{
    printf "index\tseed\ttrain_sqw_norm\tinference_corr\tmodel_path\tprediction_path\tplot_path\ttrain_metrics_path\n"
} > "${SUMMARY_PATH}"

echo "RUN_LABEL=${RUN_LABEL}"
echo "MODEL_COUNT=${MODEL_COUNT}"
echo "BASE_SEED=${BASE_SEED}"
echo "DELTA_FRAMES=${DELTA_FRAMES}"
echo "EPOCHS=${EPOCHS}"
echo "TRAIN_DATA_PATH=${TRAIN_DATA_PATH}"
echo "INFERENCE_DATA_PATH=${INFERENCE_DATA_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "MODELS_DIR=${MODELS_DIR}"

for INDEX in $(seq 0 $((MODEL_COUNT - 1))); do
    SEED=$((BASE_SEED + INDEX))
    TRAIN_METRICS_PATH="${OUTPUT_ROOT}/training_metrics_${INDEX}.tsv"
    TRAIN_PLOT_DIR="${OUTPUT_ROOT}/training_plots_${INDEX}"
    PREDICTION_PATH="${OUTPUT_ROOT}/inference_1055_${INDEX}.npz"
    PLOT_PATH="${OUTPUT_ROOT}/sqw_1055_${INDEX}.png"
    PLOT_METRICS_PATH="${OUTPUT_ROOT}/sqw_1055_${INDEX}.txt"

    echo "=== MODEL ${INDEX}/${MODEL_COUNT} seed=${SEED}: train ==="
    python find_field_rnn_models.py 1 RNN \
        --data-path "${TRAIN_DATA_PATH}" \
        --models-dir "${MODELS_DIR}" \
        --metrics-path "${TRAIN_METRICS_PATH}" \
        --count-steps "${TRAIN_COUNT_STEPS}" \
        --count-run "${COUNT_RUN}" \
        --velocity-score-weight "${VELOCITY_SCORE_WEIGHT:-0.0}" \
        --velocity-window-frames "${VELOCITY_WINDOW_FRAMES:-10}" \
        --velocity-hist-bins "${VELOCITY_HIST_BINS:-80}" \
        --velocity-max-end-speed-ratio "${VELOCITY_MAX_END_SPEED_RATIO:-inf}" \
        --delta-frames "${DELTA_FRAMES}" \
        --encoder-channels 32 \
        --rnn-hidden-size 64 \
        --rnn-layers 1 \
        --conv-layers 2 \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --data-len "${DATA_LEN}" \
        --learning-rate 0.001 \
        --target-mode acceleration \
        --acceleration-normalization channel \
        --loss-region all \
        --no-cyclic-shift-augmentation \
        --input-transform absolute \
        --input-relative-scale 1.0 \
        --force-balance-loss-weight "${FORCE_BALANCE_LOSS_WEIGHT:-0.0}" \
        --acceleration-rms-loss-weight "${ACCELERATION_RMS_LOSS_WEIGHT:-0.0}" \
        --velocity-rms-loss-weight "${VELOCITY_RMS_LOSS_WEIGHT:-0.0}" \
        --rms-loss-epsilon "${RMS_LOSS_EPSILON:-1e-12}" \
        --device "${DEVICE}" \
        --bidirectional \
        --save-all \
        --plot-all \
        --plot-output-dir "${TRAIN_PLOT_DIR}" \
        --seed "${SEED}"

    MODEL_PATH="$(awk -F '\t' 'NR == 2 {print $2}' "${TRAIN_METRICS_PATH}")"
    TRAIN_SQW_NORM="$(awk -F '\t' 'NR == 2 {print $3}' "${TRAIN_METRICS_PATH}")"
    if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
        echo "Training did not produce a usable model path for index ${INDEX}: ${MODEL_PATH}" >&2
        exit 3
    fi

    echo "=== MODEL ${INDEX}/${MODEL_COUNT} seed=${SEED}: inference on 1055 ==="
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "TRAIN_SQW_NORM=${TRAIN_SQW_NORM}"
    python infer_field_rnn_centered_acceleration.py \
        --model-path "${MODEL_PATH}" \
        --data-path "${INFERENCE_DATA_PATH}" \
        --output-path "${PREDICTION_PATH}" \
        --count-steps "${INFERENCE_COUNT_STEPS}" \
        --start-frame "${START_FRAME}" \
        --patch-shape 3 3 3 \
        --patch-batch-size "${PATCH_BATCH_SIZE}" \
        --velocity-thermostat-interval "${VELOCITY_THERMOSTAT_INTERVAL:-0}" \
        --velocity-thermostat-coupling "${VELOCITY_THERMOSTAT_COUPLING:-1.0}" \
        --device "${DEVICE}" \
        --reference-output \
        --save-positions

    echo "=== MODEL ${INDEX}/${MODEL_COUNT} seed=${SEED}: S(q,w) plot ==="
    python plot_sqw_comparison.py \
        --input-path "${PREDICTION_PATH}" \
        --output-path "${PLOT_PATH}" \
        --ncells "${NCELLS}" \
        --kcount "${KCOUNT}" | tee "${PLOT_METRICS_PATH}"

    INFERENCE_CORR="$(awk '/correlation =/ {print $3}' "${PLOT_METRICS_PATH}")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${INDEX}" \
        "${SEED}" \
        "${TRAIN_SQW_NORM}" \
        "${INFERENCE_CORR}" \
        "${MODEL_PATH}" \
        "${PREDICTION_PATH}" \
        "${PLOT_PATH}" \
        "${TRAIN_METRICS_PATH}" >> "${SUMMARY_PATH}"

    echo "=== MODEL ${INDEX}/${MODEL_COUNT} done: train=${TRAIN_SQW_NORM}, corr1055=${INFERENCE_CORR} ==="
done

python - "${SUMMARY_PATH}" "${MONTAGE_PATH}" <<'PY'
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

summary_path = Path(sys.argv[1])
montage_path = Path(sys.argv[2])

with summary_path.open(newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

if not rows:
    raise SystemExit("No rows in summary; cannot create montage")

cols = 2
rows_count = (len(rows) + cols - 1) // cols
fig, axes = plt.subplots(rows_count, cols, figsize=(14, 4.8 * rows_count), constrained_layout=True)
axes = axes.reshape(rows_count, cols)

for ax in axes.ravel():
    ax.axis("off")

for row, ax in zip(rows, axes.ravel()):
    image = mpimg.imread(row["plot_path"])
    ax.imshow(image)
    ax.set_title(
        f"#{row['index']} seed={row['seed']} | "
        f"333 score={float(row['train_sqw_norm']):.4f} | "
        f"1055 corr={float(row['inference_corr']):.4f}",
        fontsize=10,
    )
    ax.axis("off")

montage_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(montage_path, dpi=180)
plt.close(fig)
print(f"Saved montage {montage_path}")
PY

echo "SUMMARY_PATH=${SUMMARY_PATH}"
echo "MONTAGE_PATH=${MONTAGE_PATH}"
echo "DONE"
