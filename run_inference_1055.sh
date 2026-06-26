#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

set +u
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
set -u

MODEL_PATH="${MODEL_PATH:-models333/mean_norm_1.7796845622334774_rrn_crystal_400.pth}"
DATA_PATH="${DATA_PATH:-data1055.npz}"
START_FRAME="${START_FRAME:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-inference_outputs}"
SQW_STEP="${SQW_STEP:-5}"

read -r SEQUENCE_LENGTH FRAME_COUNT <<< "$(
python - "$DATA_PATH" <<'PY'
import numpy as np
import sys

data = np.load(sys.argv[1])
print(int(data["X_blocks"].shape[1]), int(data["displacements"].shape[0]))
PY
)"

MAX_COUNT_STEPS=$((FRAME_COUNT - SEQUENCE_LENGTH - START_FRAME))
if (( MAX_COUNT_STEPS <= 0 )); then
  echo "Not enough frames for START_FRAME=$START_FRAME and sequence_length=$SEQUENCE_LENGTH" >&2
  exit 1
fi

COUNT_STEPS="${COUNT_STEPS:-$MAX_COUNT_STEPS}"
if (( COUNT_STEPS > MAX_COUNT_STEPS )); then
  echo "COUNT_STEPS=$COUNT_STEPS is too large for reference output." >&2
  echo "Maximum with DATA_PATH=$DATA_PATH and START_FRAME=$START_FRAME is $MAX_COUNT_STEPS." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

BASE_NAME="data1055_start_${START_FRAME}_steps_${COUNT_STEPS}"
PREDICTION_PATH="$OUTPUT_DIR/${BASE_NAME}_prediction.npz"
PLOT_PATH="$OUTPUT_DIR/${BASE_NAME}_sqw.png"

python infer_model.py \
  --model-path "$MODEL_PATH" \
  --data-path "$DATA_PATH" \
  --output-path "$PREDICTION_PATH" \
  --count-steps "$COUNT_STEPS" \
  --start-frame "$START_FRAME" \
  --reference-output \
  --save-positions

MPLBACKEND="${MPLBACKEND:-Agg}" MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" python plot_sqw_comparison.py \
  --input-path "$PREDICTION_PATH" \
  --output-path "$PLOT_PATH" \
  --step "$SQW_STEP"

echo "Prediction saved to $PREDICTION_PATH"
echo "S(q,w) comparison saved to $PLOT_PATH"
