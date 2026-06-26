#!/bin/bash
#SBATCH --job-name=ase1055-nvt
#SBATCH --partition=gold-batch
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ase1055_nvt_%j.out
#SBATCH --error=logs/ase1055_nvt_%j.err

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

MODEL_PATH="${MODEL_PATH:-models333_field_rnn_cl2_d30k_velocity_w1_100/mean_norm_0.354316703772241_rnn_field_rnn_acceleration_ec32_rh64_rl1_bidir_accnormchannel_cl2_k3.pth}"
DATA_PATH="${DATA_PATH:-data1055.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-inference_outputs/ase_nvt_1055/best_velocity_w1_tau200_dt002_2000}"
ASE_OUTPUT_PATH="${OUTPUT_ROOT}/ase_nvt_1055.npz"
ASE_LOG_PATH="${OUTPUT_ROOT}/ase_nvt_1055.log"
PLOT_INPUT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_plot_input.npz"
SQW_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_sqw.png"
SQW_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_sqw.txt"
VELOCITY_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_velocity_histograms.png"
VELOCITY_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_velocity_histograms.txt"
DISPLACEMENT_RAW_HIST_PATH="${OUTPUT_ROOT}/ase_nvt_1055_displacement_components_raw.png"
DISPLACEMENT_RAW_LAYER_PATH="${OUTPUT_ROOT}/ase_nvt_1055_displacement_layers_raw.png"
DISPLACEMENT_RAW_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_displacement_components_raw.txt"
DISPLACEMENT_COMSUB_HIST_PATH="${OUTPUT_ROOT}/ase_nvt_1055_displacement_components_comsub.png"
DISPLACEMENT_COMSUB_LAYER_PATH="${OUTPUT_ROOT}/ase_nvt_1055_displacement_layers_comsub.png"
DISPLACEMENT_COMSUB_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_displacement_components_comsub.txt"
VACF_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_vacf.png"
VACF_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_vacf.tsv"
RDF_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_rdf.png"
RDF_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_rdf.tsv"
TEMPERATURE_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_temperature_trace.png"
TEMPERATURE_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_temperature_trace.txt"
COM_MOTION_PREFIX="${OUTPUT_ROOT}/ase_nvt_1055_com_motion"
COM_MOTION_PLOT_PATH="${COM_MOTION_PREFIX}.png"
COM_MOTION_METRICS_PATH="${COM_MOTION_PREFIX}.txt"
PHASE_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_phase_power_acceleration.png"
PHASE_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_phase_power_acceleration.tsv"
CANONICAL_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_canonical_checks.png"
CANONICAL_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_canonical_checks.tsv"
SOUND_SPEED_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_sound_speed_ox.png"
SOUND_SPEED_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_sound_speed_ox.tsv"
PATH_ENERGY_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_path_energy.png"
PATH_ENERGY_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_path_energy.tsv"
ETOT_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_etot_trace.png"
ETOT_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_etot_trace.tsv"
ETOT_CUMULATIVE_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_etot_cumulative.png"
ETOT_CUMULATIVE_SERIES_PATH="${OUTPUT_ROOT}/ase_nvt_1055_etot_cumulative.tsv"
ETOT_CUMULATIVE_SUMMARY_PATH="${OUTPUT_ROOT}/ase_nvt_1055_etot_cumulative_summary.tsv"
HEAT_CAPACITY_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_heat_capacity.png"
HEAT_CAPACITY_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_heat_capacity.tsv"
ETOT_HEAT_CAPACITY_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_etot_heat_capacity.png"
ETOT_HEAT_CAPACITY_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_etot_heat_capacity.tsv"

STEPS="${STEPS:-2000}"
DT_PS="${DT_PS:-0.002}"
TEMPERATURE_K="${TEMPERATURE_K:-300}"
TAUT_FS="${TAUT_FS:-200}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-250}"
DEVICE="${DEVICE:-cuda}"
NCELLS="${NCELLS:-10}"
KCOUNT="${KCOUNT:-10}"
SQW_STEP="${SQW_STEP:-10}"
VELOCITY_WINDOW="${VELOCITY_WINDOW:-10}"
VELOCITY_BINS="${VELOCITY_BINS:-120}"
DISPLACEMENT_BINS="${DISPLACEMENT_BINS:-180}"
DISPLACEMENT_PERCENTILE="${DISPLACEMENT_PERCENTILE:-99.5}"
DISPLACEMENT_CHUNK_FRAMES="${DISPLACEMENT_CHUNK_FRAMES:-512}"
DISPLACEMENT_SAMPLE_FRAME_STRIDE="${DISPLACEMENT_SAMPLE_FRAME_STRIDE:-20}"
VACF_MAX_LAG_FRAMES="${VACF_MAX_LAG_FRAMES:-2500}"
VACF_FFT_CHUNK_SIZE="${VACF_FFT_CHUNK_SIZE:-512}"
VACF_REMOVE_FRAME_COM="${VACF_REMOVE_FRAME_COM:-1}"
RDF_BINS="${RDF_BINS:-220}"
RDF_MAX_FRAMES="${RDF_MAX_FRAMES:-500}"
RDF_R_MAX="${RDF_R_MAX:-}"
RDF_FRAME_FRACTION_START="${RDF_FRAME_FRACTION_START:-0.0}"
RDF_FRAME_FRACTION_END="${RDF_FRAME_FRACTION_END:-1.0}"
PHASE_WINDOW_FRAMES="${PHASE_WINDOW_FRAMES:-1000}"
PHASE_BINS="${PHASE_BINS:-120}"
SOUND_FRAME_STRIDE="${SOUND_FRAME_STRIDE:-1}"
SOUND_FIT_Q_COUNT="${SOUND_FIT_Q_COUNT:-3}"
SOUND_Q_INDEX_MIN="${SOUND_Q_INDEX_MIN:-1}"
SOUND_Q_INDEX_MAX="${SOUND_Q_INDEX_MAX:-}"
SOUND_MIN_ENERGY_MEV="${SOUND_MIN_ENERGY_MEV:-0.25}"
SOUND_MAX_ENERGY_MEV="${SOUND_MAX_ENERGY_MEV:-45.0}"
SOUND_SMOOTH_BINS="${SOUND_SMOOTH_BINS:-5}"
PATH_ENERGY_SLABS="${PATH_ENERGY_SLABS:-10}"
PATH_ENERGY_PLOT_STRIDE="${PATH_ENERGY_PLOT_STRIDE:-1}"
PATH_ENERGY_FIT_FRACTION_START="${PATH_ENERGY_FIT_FRACTION_START:-0.1}"
PATH_ENERGY_FIT_FRACTION_END="${PATH_ENERGY_FIT_FRACTION_END:-1.0}"
HEAT_CAPACITY_FRAME_FRACTION_START="${HEAT_CAPACITY_FRAME_FRACTION_START:-0.0}"
HEAT_CAPACITY_FRAME_FRACTION_END="${HEAT_CAPACITY_FRAME_FRACTION_END:-1.0}"
HEAT_CAPACITY_DETREND="${HEAT_CAPACITY_DETREND:-1}"
HEAT_CAPACITY_BLOCK_SIZES="${HEAT_CAPACITY_BLOCK_SIZES:-250 500 1000 2500 5000}"
Q_ZERO_MODE="${Q_ZERO_MODE:-initial}"
LOW_Q_CORRECTION_MODE="${LOW_Q_CORRECTION_MODE:-none}"
LOW_Q_CORRECTION_MAX_Q="${LOW_Q_CORRECTION_MAX_Q:-0.0}"
LOW_Q_CORRECTION_BLEND="${LOW_Q_CORRECTION_BLEND:-1.0}"
LOW_Q_CORRECTION_EXCLUDE_Q_ZERO="${LOW_Q_CORRECTION_EXCLUDE_Q_ZERO:-1}"
LOW_Q_REFERENCE_MAX_FRAMES="${LOW_Q_REFERENCE_MAX_FRAMES:-}"
LOW_Q_STIFFNESS_EPSILON="${LOW_Q_STIFFNESS_EPSILON:-1e-30}"
ACCELERATION_SCALE="${ACCELERATION_SCALE:-1.0}"
CURL_CORRECTION_MODE="${CURL_CORRECTION_MODE:-none}"
CURL_CORRECTION_ETA="${CURL_CORRECTION_ETA:-0.0}"
CURL_CORRECTION_INTERVAL="${CURL_CORRECTION_INTERVAL:-10}"
CURL_CORRECTION_BATCH_SIZE="${CURL_CORRECTION_BATCH_SIZE:-32}"
CURL_CORRECTION_EPSILON="${CURL_CORRECTION_EPSILON:-1e-12}"
HISTORY_DAMPING_MODE="${HISTORY_DAMPING_MODE:-none}"
HISTORY_DAMPING_ETA="${HISTORY_DAMPING_ETA:-0.0}"
HISTORY_DAMPING_INTERVAL="${HISTORY_DAMPING_INTERVAL:-10}"
HISTORY_DAMPING_BATCH_SIZE="${HISTORY_DAMPING_BATCH_SIZE:-32}"
HISTORY_DAMPING_ADAPTIVE_GAIN="${HISTORY_DAMPING_ADAPTIVE_GAIN:-0.0}"
HISTORY_DAMPING_ADAPTIVE_COOLING_GAIN="${HISTORY_DAMPING_ADAPTIVE_COOLING_GAIN:-}"
HISTORY_DAMPING_ADAPTIVE_INTERVAL="${HISTORY_DAMPING_ADAPTIVE_INTERVAL:-100}"
HISTORY_DAMPING_ADAPTIVE_MIN_ETA="${HISTORY_DAMPING_ADAPTIVE_MIN_ETA:-0.0}"
HISTORY_DAMPING_ADAPTIVE_MAX_ETA="${HISTORY_DAMPING_ADAPTIVE_MAX_ETA:-}"
HISTORY_DAMPING_ADAPTIVE_TARGET_POWER="${HISTORY_DAMPING_ADAPTIVE_TARGET_POWER:-0.0}"
HISTORY_DAMPING_ADAPTIVE_EMA="${HISTORY_DAMPING_ADAPTIVE_EMA:-0.05}"
HISTORY_DAMPING_ADAPTIVE_EPSILON="${HISTORY_DAMPING_ADAPTIVE_EPSILON:-1e-30}"
TEMPERATURE_ETA_ADAPTIVE_MODE="${TEMPERATURE_ETA_ADAPTIVE_MODE:-none}"
TEMPERATURE_ETA_ADAPTIVE_GAIN="${TEMPERATURE_ETA_ADAPTIVE_GAIN:-0.0}"
TEMPERATURE_ETA_ADAPTIVE_INTERVAL="${TEMPERATURE_ETA_ADAPTIVE_INTERVAL:-100}"
TEMPERATURE_ETA_ADAPTIVE_EMA="${TEMPERATURE_ETA_ADAPTIVE_EMA:-0.002}"
TEMPERATURE_ETA_ADAPTIVE_MIN_ETA="${TEMPERATURE_ETA_ADAPTIVE_MIN_ETA:-0.0}"
TEMPERATURE_ETA_ADAPTIVE_MAX_ETA="${TEMPERATURE_ETA_ADAPTIVE_MAX_ETA:-}"
TEMPERATURE_ETA_ADAPTIVE_DEADBAND="${TEMPERATURE_ETA_ADAPTIVE_DEADBAND:-0.0}"
POWER_BIAS_CORRECTION_MODE="${POWER_BIAS_CORRECTION_MODE:-none}"
POWER_BIAS_CORRECTION_ALPHA="${POWER_BIAS_CORRECTION_ALPHA:-1.0}"
POWER_BIAS_CORRECTION_EPSILON="${POWER_BIAS_CORRECTION_EPSILON:-1e-30}"

LOW_Q_EXCLUDE_FLAG="--low-q-correction-exclude-q-zero"
if [[ "${LOW_Q_CORRECTION_EXCLUDE_Q_ZERO}" == "0" || "${LOW_Q_CORRECTION_EXCLUDE_Q_ZERO}" == "false" ]]; then
    LOW_Q_EXCLUDE_FLAG="--no-low-q-correction-exclude-q-zero"
fi

LOW_Q_REFERENCE_MAX_FRAMES_ARGS=()
if [[ -n "${LOW_Q_REFERENCE_MAX_FRAMES}" ]]; then
    LOW_Q_REFERENCE_MAX_FRAMES_ARGS=(--low-q-reference-max-frames "${LOW_Q_REFERENCE_MAX_FRAMES}")
fi

HISTORY_DAMPING_ADAPTIVE_MAX_ETA_ARGS=()
if [[ -n "${HISTORY_DAMPING_ADAPTIVE_MAX_ETA}" ]]; then
    HISTORY_DAMPING_ADAPTIVE_MAX_ETA_ARGS=(
        --history-damping-adaptive-max-eta "${HISTORY_DAMPING_ADAPTIVE_MAX_ETA}"
    )
fi

HISTORY_DAMPING_ADAPTIVE_COOLING_GAIN_ARGS=()
if [[ -n "${HISTORY_DAMPING_ADAPTIVE_COOLING_GAIN}" ]]; then
    HISTORY_DAMPING_ADAPTIVE_COOLING_GAIN_ARGS=(
        --history-damping-adaptive-cooling-gain "${HISTORY_DAMPING_ADAPTIVE_COOLING_GAIN}"
    )
fi

TEMPERATURE_ETA_ADAPTIVE_MAX_ETA_ARGS=()
if [[ -n "${TEMPERATURE_ETA_ADAPTIVE_MAX_ETA}" ]]; then
    TEMPERATURE_ETA_ADAPTIVE_MAX_ETA_ARGS=(
        --temperature-eta-adaptive-max-eta "${TEMPERATURE_ETA_ADAPTIVE_MAX_ETA}"
    )
fi

SOUND_Q_INDEX_MAX_ARGS=()
if [[ -n "${SOUND_Q_INDEX_MAX}" ]]; then
    SOUND_Q_INDEX_MAX_ARGS=(--q-index-max "${SOUND_Q_INDEX_MAX}")
fi

SOUND_MAX_ENERGY_ARGS=()
if [[ -n "${SOUND_MAX_ENERGY_MEV}" ]]; then
    SOUND_MAX_ENERGY_ARGS=(--max-energy-mev "${SOUND_MAX_ENERGY_MEV}")
fi

HEAT_CAPACITY_DETREND_FLAG="--detrend"
if [[ "${HEAT_CAPACITY_DETREND}" == "0" || "${HEAT_CAPACITY_DETREND}" == "false" ]]; then
    HEAT_CAPACITY_DETREND_FLAG="--no-detrend"
fi

VACF_REMOVE_FRAME_COM_FLAG="--remove-frame-com"
if [[ "${VACF_REMOVE_FRAME_COM}" == "0" || "${VACF_REMOVE_FRAME_COM}" == "false" ]]; then
    VACF_REMOVE_FRAME_COM_FLAG="--no-remove-frame-com"
fi

RDF_R_MAX_ARGS=()
if [[ -n "${RDF_R_MAX}" ]]; then
    RDF_R_MAX_ARGS=(--r-max "${RDF_R_MAX}")
fi

mkdir -p "${OUTPUT_ROOT}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_PATH=${DATA_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "STEPS=${STEPS}"
echo "DT_PS=${DT_PS}"
echo "TEMPERATURE_K=${TEMPERATURE_K}"
echo "TAUT_FS=${TAUT_FS}"
echo "Q_ZERO_MODE=${Q_ZERO_MODE}"
echo "LOW_Q_CORRECTION_MODE=${LOW_Q_CORRECTION_MODE}"
echo "LOW_Q_CORRECTION_MAX_Q=${LOW_Q_CORRECTION_MAX_Q}"
echo "LOW_Q_CORRECTION_BLEND=${LOW_Q_CORRECTION_BLEND}"
echo "LOW_Q_CORRECTION_EXCLUDE_Q_ZERO=${LOW_Q_CORRECTION_EXCLUDE_Q_ZERO}"
echo "LOW_Q_REFERENCE_MAX_FRAMES=${LOW_Q_REFERENCE_MAX_FRAMES}"
echo "ACCELERATION_SCALE=${ACCELERATION_SCALE}"
echo "CURL_CORRECTION_MODE=${CURL_CORRECTION_MODE}"
echo "CURL_CORRECTION_ETA=${CURL_CORRECTION_ETA}"
echo "CURL_CORRECTION_INTERVAL=${CURL_CORRECTION_INTERVAL}"
echo "CURL_CORRECTION_BATCH_SIZE=${CURL_CORRECTION_BATCH_SIZE}"
echo "HISTORY_DAMPING_MODE=${HISTORY_DAMPING_MODE}"
echo "HISTORY_DAMPING_ETA=${HISTORY_DAMPING_ETA}"
echo "HISTORY_DAMPING_INTERVAL=${HISTORY_DAMPING_INTERVAL}"
echo "HISTORY_DAMPING_BATCH_SIZE=${HISTORY_DAMPING_BATCH_SIZE}"
echo "HISTORY_DAMPING_ADAPTIVE_GAIN=${HISTORY_DAMPING_ADAPTIVE_GAIN}"
echo "HISTORY_DAMPING_ADAPTIVE_COOLING_GAIN=${HISTORY_DAMPING_ADAPTIVE_COOLING_GAIN}"
echo "HISTORY_DAMPING_ADAPTIVE_INTERVAL=${HISTORY_DAMPING_ADAPTIVE_INTERVAL}"
echo "HISTORY_DAMPING_ADAPTIVE_MIN_ETA=${HISTORY_DAMPING_ADAPTIVE_MIN_ETA}"
echo "HISTORY_DAMPING_ADAPTIVE_MAX_ETA=${HISTORY_DAMPING_ADAPTIVE_MAX_ETA}"
echo "HISTORY_DAMPING_ADAPTIVE_TARGET_POWER=${HISTORY_DAMPING_ADAPTIVE_TARGET_POWER}"
echo "HISTORY_DAMPING_ADAPTIVE_EMA=${HISTORY_DAMPING_ADAPTIVE_EMA}"
echo "TEMPERATURE_ETA_ADAPTIVE_MODE=${TEMPERATURE_ETA_ADAPTIVE_MODE}"
echo "TEMPERATURE_ETA_ADAPTIVE_GAIN=${TEMPERATURE_ETA_ADAPTIVE_GAIN}"
echo "TEMPERATURE_ETA_ADAPTIVE_INTERVAL=${TEMPERATURE_ETA_ADAPTIVE_INTERVAL}"
echo "TEMPERATURE_ETA_ADAPTIVE_EMA=${TEMPERATURE_ETA_ADAPTIVE_EMA}"
echo "TEMPERATURE_ETA_ADAPTIVE_MIN_ETA=${TEMPERATURE_ETA_ADAPTIVE_MIN_ETA}"
echo "TEMPERATURE_ETA_ADAPTIVE_MAX_ETA=${TEMPERATURE_ETA_ADAPTIVE_MAX_ETA}"
echo "TEMPERATURE_ETA_ADAPTIVE_DEADBAND=${TEMPERATURE_ETA_ADAPTIVE_DEADBAND}"
echo "POWER_BIAS_CORRECTION_MODE=${POWER_BIAS_CORRECTION_MODE}"
echo "POWER_BIAS_CORRECTION_ALPHA=${POWER_BIAS_CORRECTION_ALPHA}"
echo "POWER_BIAS_CORRECTION_EPSILON=${POWER_BIAS_CORRECTION_EPSILON}"
echo "DISPLACEMENT_BINS=${DISPLACEMENT_BINS}"
echo "DISPLACEMENT_PERCENTILE=${DISPLACEMENT_PERCENTILE}"
echo "DISPLACEMENT_CHUNK_FRAMES=${DISPLACEMENT_CHUNK_FRAMES}"
echo "DISPLACEMENT_SAMPLE_FRAME_STRIDE=${DISPLACEMENT_SAMPLE_FRAME_STRIDE}"
echo "PHASE_WINDOW_FRAMES=${PHASE_WINDOW_FRAMES}"
echo "PHASE_BINS=${PHASE_BINS}"
echo "SOUND_FRAME_STRIDE=${SOUND_FRAME_STRIDE}"
echo "SOUND_FIT_Q_COUNT=${SOUND_FIT_Q_COUNT}"
echo "SOUND_Q_INDEX_MIN=${SOUND_Q_INDEX_MIN}"
echo "SOUND_Q_INDEX_MAX=${SOUND_Q_INDEX_MAX}"
echo "SOUND_MIN_ENERGY_MEV=${SOUND_MIN_ENERGY_MEV}"
echo "SOUND_MAX_ENERGY_MEV=${SOUND_MAX_ENERGY_MEV}"
echo "SOUND_SMOOTH_BINS=${SOUND_SMOOTH_BINS}"
echo "PATH_ENERGY_SLABS=${PATH_ENERGY_SLABS}"
echo "PATH_ENERGY_PLOT_STRIDE=${PATH_ENERGY_PLOT_STRIDE}"
echo "PATH_ENERGY_FIT_FRACTION_START=${PATH_ENERGY_FIT_FRACTION_START}"
echo "PATH_ENERGY_FIT_FRACTION_END=${PATH_ENERGY_FIT_FRACTION_END}"
echo "HEAT_CAPACITY_FRAME_FRACTION_START=${HEAT_CAPACITY_FRAME_FRACTION_START}"
echo "HEAT_CAPACITY_FRAME_FRACTION_END=${HEAT_CAPACITY_FRAME_FRACTION_END}"
echo "HEAT_CAPACITY_DETREND=${HEAT_CAPACITY_DETREND}"
echo "HEAT_CAPACITY_BLOCK_SIZES=${HEAT_CAPACITY_BLOCK_SIZES}"
echo "VACF_MAX_LAG_FRAMES=${VACF_MAX_LAG_FRAMES}"
echo "VACF_FFT_CHUNK_SIZE=${VACF_FFT_CHUNK_SIZE}"
echo "VACF_REMOVE_FRAME_COM=${VACF_REMOVE_FRAME_COM}"
echo "RDF_BINS=${RDF_BINS}"
echo "RDF_MAX_FRAMES=${RDF_MAX_FRAMES}"
echo "RDF_R_MAX=${RDF_R_MAX}"
echo "RDF_FRAME_FRACTION_START=${RDF_FRAME_FRACTION_START}"
echo "RDF_FRAME_FRACTION_END=${RDF_FRAME_FRACTION_END}"
echo "DEVICE=${DEVICE}"

python run_ase_copper_nvt.py \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --output-npz "${ASE_OUTPUT_PATH}" \
    --log-path "${ASE_LOG_PATH}" \
    --steps "${STEPS}" \
    --initial-frames 0 1 2 \
    --temperature-k "${TEMPERATURE_K}" \
    --taut-fs "${TAUT_FS}" \
    --dt-ps "${DT_PS}" \
    --patch-shape 3 3 3 \
    --patch-batch-size "${PATCH_BATCH_SIZE}" \
    --device "${DEVICE}" \
    --q-zero-mode "${Q_ZERO_MODE}" \
    --low-q-correction-mode "${LOW_Q_CORRECTION_MODE}" \
    --low-q-correction-max-q "${LOW_Q_CORRECTION_MAX_Q}" \
    --low-q-correction-blend "${LOW_Q_CORRECTION_BLEND}" \
    "${LOW_Q_EXCLUDE_FLAG}" \
    "${LOW_Q_REFERENCE_MAX_FRAMES_ARGS[@]}" \
    --low-q-stiffness-epsilon "${LOW_Q_STIFFNESS_EPSILON}" \
    --acceleration-scale "${ACCELERATION_SCALE}" \
    --curl-correction-mode "${CURL_CORRECTION_MODE}" \
    --curl-correction-eta "${CURL_CORRECTION_ETA}" \
    --curl-correction-interval "${CURL_CORRECTION_INTERVAL}" \
    --curl-correction-batch-size "${CURL_CORRECTION_BATCH_SIZE}" \
    --curl-correction-epsilon "${CURL_CORRECTION_EPSILON}" \
    --history-damping-mode "${HISTORY_DAMPING_MODE}" \
    --history-damping-eta "${HISTORY_DAMPING_ETA}" \
    --history-damping-interval "${HISTORY_DAMPING_INTERVAL}" \
    --history-damping-batch-size "${HISTORY_DAMPING_BATCH_SIZE}" \
    --history-damping-adaptive-gain "${HISTORY_DAMPING_ADAPTIVE_GAIN}" \
    "${HISTORY_DAMPING_ADAPTIVE_COOLING_GAIN_ARGS[@]}" \
    --history-damping-adaptive-interval "${HISTORY_DAMPING_ADAPTIVE_INTERVAL}" \
    --history-damping-adaptive-min-eta "${HISTORY_DAMPING_ADAPTIVE_MIN_ETA}" \
    "${HISTORY_DAMPING_ADAPTIVE_MAX_ETA_ARGS[@]}" \
    --history-damping-adaptive-target-power "${HISTORY_DAMPING_ADAPTIVE_TARGET_POWER}" \
    --history-damping-adaptive-ema "${HISTORY_DAMPING_ADAPTIVE_EMA}" \
    --history-damping-adaptive-epsilon "${HISTORY_DAMPING_ADAPTIVE_EPSILON}" \
    --temperature-eta-adaptive-mode "${TEMPERATURE_ETA_ADAPTIVE_MODE}" \
    --temperature-eta-adaptive-gain "${TEMPERATURE_ETA_ADAPTIVE_GAIN}" \
    --temperature-eta-adaptive-interval "${TEMPERATURE_ETA_ADAPTIVE_INTERVAL}" \
    --temperature-eta-adaptive-ema "${TEMPERATURE_ETA_ADAPTIVE_EMA}" \
    --temperature-eta-adaptive-min-eta "${TEMPERATURE_ETA_ADAPTIVE_MIN_ETA}" \
    "${TEMPERATURE_ETA_ADAPTIVE_MAX_ETA_ARGS[@]}" \
    --temperature-eta-adaptive-deadband "${TEMPERATURE_ETA_ADAPTIVE_DEADBAND}" \
    --power-bias-correction-mode "${POWER_BIAS_CORRECTION_MODE}" \
    --power-bias-correction-alpha "${POWER_BIAS_CORRECTION_ALPHA}" \
    --power-bias-correction-epsilon "${POWER_BIAS_CORRECTION_EPSILON}" \
    --no-rescale-initial-temperature

python - <<PY
from pathlib import Path

import numpy as np

from plot_sqw_comparison import crystal_frames_to_flat_positions

ase_path = Path("${ASE_OUTPUT_PATH}")
data_path = Path("${DATA_PATH}")
out_path = Path("${PLOT_INPUT_PATH}")

ase = np.load(ase_path)
data = np.load(data_path)
positions = ase["positions"].astype(np.float32)
start = int(ase["initial_frames"][-1])
available_reference_frames = max(0, int(data["displacements"].shape[0]) - start)
requested_frames = int(positions.shape[0])
frames = min(requested_frames, available_reference_frames)
if frames < 2:
    raise ValueError(
        f"Not enough overlapping frames for comparison: "
        f"requested={requested_frames}, reference_available={available_reference_frames}, start={start}"
    )
if frames < requested_frames:
    print(
        f"WARNING: trimming ASE trajectory for comparison from {requested_frames} to {frames} frames "
        f"because reference data ends at frame {int(data['displacements'].shape[0]) - 1}."
    )
positions = positions[:frames]
reference_displacements = data["displacements"][start:start + frames].astype(np.float32)
reference_positions = crystal_frames_to_flat_positions(
    reference_displacements,
    data["reference_positions"],
    data["atom_order"],
)

np.savez_compressed(
    out_path,
    predicted_positions=positions.reshape(frames, -1),
    reference_positions_output=reference_positions,
    predicted_positions_ase=positions,
    reference_frame_start=np.asarray(start, dtype=np.int64),
    comparison_frames=np.asarray(frames, dtype=np.int64),
    dt_ps=ase["dt_ps"],
    temperature_k=ase["temperature_k"][:frames],
    kinetic_energy_ev=ase["kinetic_energy_ev"][:frames],
)

temperature = np.asarray(ase["temperature_k"], dtype=np.float64)
print(f"Saved {out_path}")
print(
    "temperature first/last/mean/min/max = "
    f"{temperature[0]:.8g} {temperature[-1]:.8g} "
    f"{temperature.mean():.8g} {temperature.min():.8g} {temperature.max():.8g}"
)
PY

python plot_sqw_comparison.py \
    --input-path "${PLOT_INPUT_PATH}" \
    --output-path "${SQW_PLOT_PATH}" \
    --ncells "${NCELLS}" \
    --kcount "${KCOUNT}" \
    --dt "${DT_PS}" \
    --step "${SQW_STEP}" | tee "${SQW_METRICS_PATH}"

python plot_ase_velocity_histograms.py \
    --ase-path "${ASE_OUTPUT_PATH}" \
    --data-path "${DATA_PATH}" \
    --output-path "${VELOCITY_PLOT_PATH}" \
    --metrics-path "${VELOCITY_METRICS_PATH}" \
    --window-frames "${VELOCITY_WINDOW}" \
    --bins "${VELOCITY_BINS}" \
    --title "ASE NVT velocity histograms: full inference vs available reference"

python plot_displacement_component_layers.py \
    --ase-path "${ASE_OUTPUT_PATH}" \
    --data-path "${DATA_PATH}" \
    --hist-output-path "${DISPLACEMENT_RAW_HIST_PATH}" \
    --layer-output-path "${DISPLACEMENT_RAW_LAYER_PATH}" \
    --metrics-path "${DISPLACEMENT_RAW_METRICS_PATH}" \
    --bins "${DISPLACEMENT_BINS}" \
    --percentile "${DISPLACEMENT_PERCENTILE}" \
    --chunk-frames "${DISPLACEMENT_CHUNK_FRAMES}" \
    --sample-frame-stride "${DISPLACEMENT_SAMPLE_FRAME_STRIDE}" \
    --title "ASE NVT displacement diagnostics: raw"

python plot_displacement_component_layers.py \
    --ase-path "${ASE_OUTPUT_PATH}" \
    --data-path "${DATA_PATH}" \
    --hist-output-path "${DISPLACEMENT_COMSUB_HIST_PATH}" \
    --layer-output-path "${DISPLACEMENT_COMSUB_LAYER_PATH}" \
    --metrics-path "${DISPLACEMENT_COMSUB_METRICS_PATH}" \
    --bins "${DISPLACEMENT_BINS}" \
    --percentile "${DISPLACEMENT_PERCENTILE}" \
    --chunk-frames "${DISPLACEMENT_CHUNK_FRAMES}" \
    --sample-frame-stride "${DISPLACEMENT_SAMPLE_FRAME_STRIDE}" \
    --subtract-frame-mean \
    --title "ASE NVT displacement diagnostics: COM-subtracted"

python plot_ase_vacf.py \
    --ase-path "${ASE_OUTPUT_PATH}" \
    --data-path "${DATA_PATH}" \
    --output-path "${VACF_PLOT_PATH}" \
    --metrics-path "${VACF_METRICS_PATH}" \
    --max-lag-frames "${VACF_MAX_LAG_FRAMES}" \
    --fft-chunk-size "${VACF_FFT_CHUNK_SIZE}" \
    "${VACF_REMOVE_FRAME_COM_FLAG}" \
    --title "ASE NVT velocity autocorrelation: inference vs reference"

python plot_ase_rdf.py \
    --ase-path "${ASE_OUTPUT_PATH}" \
    --data-path "${DATA_PATH}" \
    --output-path "${RDF_PLOT_PATH}" \
    --metrics-path "${RDF_METRICS_PATH}" \
    --bins "${RDF_BINS}" \
    --max-frames "${RDF_MAX_FRAMES}" \
    "${RDF_R_MAX_ARGS[@]}" \
    --frame-fraction-start "${RDF_FRAME_FRACTION_START}" \
    --frame-fraction-end "${RDF_FRAME_FRACTION_END}" \
    --title "ASE NVT radial distribution function: inference vs reference"

python plot_ase_temperature_trace.py \
    --ase-path "${ASE_OUTPUT_PATH}" \
    --output-path "${TEMPERATURE_PLOT_PATH}" \
    --metrics-path "${TEMPERATURE_METRICS_PATH}" \
    --target-temperature-k "${TEMPERATURE_K}" \
    --title "ASE NVT temperature trace"

python plot_ase_com_motion.py \
    --ase-npz "${ASE_OUTPUT_PATH}" \
    --output-prefix "${COM_MOTION_PREFIX}"

python plot_ase_phase_histograms.py \
    --ase-path "${ASE_OUTPUT_PATH}" \
    --data-path "${DATA_PATH}" \
    --output-path "${PHASE_PLOT_PATH}" \
    --metrics-path "${PHASE_METRICS_PATH}" \
    --window-frames "${PHASE_WINDOW_FRAMES}" \
    --bins "${PHASE_BINS}" \
    --title "ASE NVT phase power and acceleration diagnostics"

python plot_ase_canonical_checks.py \
    --ase-path "${ASE_OUTPUT_PATH}" \
    --data-path "${DATA_PATH}" \
    --output-path "${CANONICAL_PLOT_PATH}" \
    --metrics-path "${CANONICAL_METRICS_PATH}" \
    --bins "${VELOCITY_BINS}" \
    --title "ASE NVT canonical checks"

python plot_ase_sound_speed_ox.py \
    --ase-path "${ASE_OUTPUT_PATH}" \
    --data-path "${DATA_PATH}" \
    --output-path "${SOUND_SPEED_PLOT_PATH}" \
    --metrics-path "${SOUND_SPEED_METRICS_PATH}" \
    --q-index-min "${SOUND_Q_INDEX_MIN}" \
    "${SOUND_Q_INDEX_MAX_ARGS[@]}" \
    --fit-q-count "${SOUND_FIT_Q_COUNT}" \
    --frame-stride "${SOUND_FRAME_STRIDE}" \
    --min-energy-mev "${SOUND_MIN_ENERGY_MEV}" \
    "${SOUND_MAX_ENERGY_ARGS[@]}" \
    --smooth-bins "${SOUND_SMOOTH_BINS}" \
    --title "ASE NVT longitudinal sound speed along OX"

python plot_ase_path_energy.py \
    --ase-path "${ASE_OUTPUT_PATH}" \
    --output-path "${PATH_ENERGY_PLOT_PATH}" \
    --metrics-path "${PATH_ENERGY_METRICS_PATH}" \
    --axis x \
    --slabs "${PATH_ENERGY_SLABS}" \
    --fit-fraction-start "${PATH_ENERGY_FIT_FRACTION_START}" \
    --fit-fraction-end "${PATH_ENERGY_FIT_FRACTION_END}" \
    --plot-stride "${PATH_ENERGY_PLOT_STRIDE}" \
    --title "ASE NVT path-energy diagnostics"

python plot_ase_heat_capacity.py \
    --ase-path "${ASE_OUTPUT_PATH}" \
    --output-path "${HEAT_CAPACITY_PLOT_PATH}" \
    --metrics-path "${HEAT_CAPACITY_METRICS_PATH}" \
    --frame-fraction-start "${HEAT_CAPACITY_FRAME_FRACTION_START}" \
    --frame-fraction-end "${HEAT_CAPACITY_FRAME_FRACTION_END}" \
    "${HEAT_CAPACITY_DETREND_FLAG}" \
    --block-sizes ${HEAT_CAPACITY_BLOCK_SIZES} \
    --title "ASE path-energy heat-capacity diagnostic"

HAS_DIRECT_ENERGY="$(python - <<PY
import numpy as np

with np.load("${ASE_OUTPUT_PATH}") as data:
    keys = set(data.files)
print(1 if {"kinetic_energy_ev", "potential_energy_ev"} <= keys else 0)
PY
)"

if [[ "${HAS_DIRECT_ENERGY}" == "1" ]]; then
    python plot_ase_total_energy.py \
        --ase-path "${ASE_OUTPUT_PATH}" \
        --output-path "${ETOT_PLOT_PATH}" \
        --metrics-path "${ETOT_METRICS_PATH}" \
        --title "ASE NVT direct E_tot diagnostics"

    python plot_ase_etot_heat_capacity.py \
        --ase-path "${ASE_OUTPUT_PATH}" \
        --output-path "${ETOT_HEAT_CAPACITY_PLOT_PATH}" \
        --metrics-path "${ETOT_HEAT_CAPACITY_METRICS_PATH}" \
        --frame-fraction-start "${HEAT_CAPACITY_FRAME_FRACTION_START}" \
        --frame-fraction-end "${HEAT_CAPACITY_FRAME_FRACTION_END}" \
        "${HEAT_CAPACITY_DETREND_FLAG}" \
        --block-sizes ${HEAT_CAPACITY_BLOCK_SIZES} \
        --title "ASE direct E_tot heat-capacity diagnostic"

    python plot_ase_etot_cumulative.py \
        --ase-path "${ASE_OUTPUT_PATH}" \
        --output-path "${ETOT_CUMULATIVE_PLOT_PATH}" \
        --series-path "${ETOT_CUMULATIVE_SERIES_PATH}" \
        --summary-path "${ETOT_CUMULATIVE_SUMMARY_PATH}" \
        --title "ASE NVT cumulative direct E_tot diagnostics"
else
    echo "Skipping direct E_tot diagnostics: ${ASE_OUTPUT_PATH} has no potential_energy_ev."
fi

echo "ASE_OUTPUT_PATH=${ASE_OUTPUT_PATH}"
echo "SQW_PLOT_PATH=${SQW_PLOT_PATH}"
echo "SQW_METRICS_PATH=${SQW_METRICS_PATH}"
echo "VELOCITY_PLOT_PATH=${VELOCITY_PLOT_PATH}"
echo "VELOCITY_METRICS_PATH=${VELOCITY_METRICS_PATH}"
echo "VACF_PLOT_PATH=${VACF_PLOT_PATH}"
echo "VACF_METRICS_PATH=${VACF_METRICS_PATH}"
echo "RDF_PLOT_PATH=${RDF_PLOT_PATH}"
echo "RDF_METRICS_PATH=${RDF_METRICS_PATH}"
echo "TEMPERATURE_PLOT_PATH=${TEMPERATURE_PLOT_PATH}"
echo "TEMPERATURE_METRICS_PATH=${TEMPERATURE_METRICS_PATH}"
echo "COM_MOTION_PLOT_PATH=${COM_MOTION_PLOT_PATH}"
echo "COM_MOTION_METRICS_PATH=${COM_MOTION_METRICS_PATH}"
echo "PHASE_PLOT_PATH=${PHASE_PLOT_PATH}"
echo "PHASE_METRICS_PATH=${PHASE_METRICS_PATH}"
echo "CANONICAL_PLOT_PATH=${CANONICAL_PLOT_PATH}"
echo "CANONICAL_METRICS_PATH=${CANONICAL_METRICS_PATH}"
echo "SOUND_SPEED_PLOT_PATH=${SOUND_SPEED_PLOT_PATH}"
echo "SOUND_SPEED_METRICS_PATH=${SOUND_SPEED_METRICS_PATH}"
echo "PATH_ENERGY_PLOT_PATH=${PATH_ENERGY_PLOT_PATH}"
echo "PATH_ENERGY_METRICS_PATH=${PATH_ENERGY_METRICS_PATH}"
echo "ETOT_PLOT_PATH=${ETOT_PLOT_PATH}"
echo "ETOT_METRICS_PATH=${ETOT_METRICS_PATH}"
echo "ETOT_CUMULATIVE_PLOT_PATH=${ETOT_CUMULATIVE_PLOT_PATH}"
echo "ETOT_CUMULATIVE_SERIES_PATH=${ETOT_CUMULATIVE_SERIES_PATH}"
echo "ETOT_CUMULATIVE_SUMMARY_PATH=${ETOT_CUMULATIVE_SUMMARY_PATH}"
echo "HEAT_CAPACITY_PLOT_PATH=${HEAT_CAPACITY_PLOT_PATH}"
echo "HEAT_CAPACITY_METRICS_PATH=${HEAT_CAPACITY_METRICS_PATH}"
echo "ETOT_HEAT_CAPACITY_PLOT_PATH=${ETOT_HEAT_CAPACITY_PLOT_PATH}"
echo "ETOT_HEAT_CAPACITY_METRICS_PATH=${ETOT_HEAT_CAPACITY_METRICS_PATH}"
