#!/bin/bash
#SBATCH --job-name=ase1055-chain
#SBATCH --partition=gold-batch
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ase1055_chain_%j.out
#SBATCH --error=logs/ase1055_chain_%j.err

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
OUTPUT_ROOT="${OUTPUT_ROOT:-inference_outputs/ase_nvt_1055_chain/best_old_tau_10_20_40_80}"
ASE_OUTPUT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_chain.npz"
ASE_LOG_PATH="${OUTPUT_ROOT}/ase_nvt_1055_chain.log"
PLOT_INPUT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_chain_plot_input.npz"
SQW_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_chain_sqw.png"
SQW_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_chain_sqw.txt"
VELOCITY_PLOT_PATH="${OUTPUT_ROOT}/ase_nvt_1055_chain_velocity_histograms.png"
VELOCITY_METRICS_PATH="${OUTPUT_ROOT}/ase_nvt_1055_chain_velocity_histograms.txt"

STEPS_PER_SEGMENT="${STEPS_PER_SEGMENT:-2000}"
TAUT_FS_CHAIN="${TAUT_FS_CHAIN:-10 20 40 80}"
DT_PS="${DT_PS:-0.002}"
TEMPERATURE_K="${TEMPERATURE_K:-300}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-250}"
DEVICE="${DEVICE:-cuda}"
NCELLS="${NCELLS:-10}"
KCOUNT="${KCOUNT:-10}"
SQW_STEP="${SQW_STEP:-10}"
VELOCITY_WINDOW="${VELOCITY_WINDOW:-10}"
VELOCITY_BINS="${VELOCITY_BINS:-120}"

read -r -a TAUT_ARRAY <<< "${TAUT_FS_CHAIN}"
mkdir -p "${OUTPUT_ROOT}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_PATH=${DATA_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "STEPS_PER_SEGMENT=${STEPS_PER_SEGMENT}"
echo "TAUT_FS_CHAIN=${TAUT_FS_CHAIN}"
echo "DT_PS=${DT_PS}"
echo "TEMPERATURE_K=${TEMPERATURE_K}"
echo "DEVICE=${DEVICE}"

python run_ase_copper_nvt_chain.py \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --output-npz "${ASE_OUTPUT_PATH}" \
    --log-path "${ASE_LOG_PATH}" \
    --steps-per-segment "${STEPS_PER_SEGMENT}" \
    --taut-fs-chain "${TAUT_ARRAY[@]}" \
    --initial-frames 0 1 2 \
    --temperature-k "${TEMPERATURE_K}" \
    --dt-ps "${DT_PS}" \
    --patch-shape 3 3 3 \
    --patch-batch-size "${PATCH_BATCH_SIZE}" \
    --device "${DEVICE}" \
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
    segment_index=ase["segment_index"][:frames],
    recorded_taut_fs=ase["recorded_taut_fs"][:frames],
)

temperature = np.asarray(ase["temperature_k"], dtype=np.float64)
print(f"Saved {out_path}")
print(
    "temperature first/last/mean/min/max = "
    f"{temperature[0]:.8g} {temperature[-1]:.8g} "
    f"{temperature.mean():.8g} {temperature.min():.8g} {temperature.max():.8g}"
)
for taut_fs in np.asarray(ase["taut_fs_chain"], dtype=np.float64):
    mask = np.asarray(ase["recorded_taut_fs"], dtype=np.float64) == taut_fs
    segment_temperature = temperature[mask]
    if segment_temperature.size:
        print(
            f"temperature segment taut_fs={taut_fs:g} first/last/mean/min/max = "
            f"{segment_temperature[0]:.8g} {segment_temperature[-1]:.8g} "
            f"{segment_temperature.mean():.8g} {segment_temperature.min():.8g} "
            f"{segment_temperature.max():.8g}"
        )
PY

python plot_sqw_comparison.py \
    --input-path "${PLOT_INPUT_PATH}" \
    --output-path "${SQW_PLOT_PATH}" \
    --ncells "${NCELLS}" \
    --kcount "${KCOUNT}" \
    --dt "${DT_PS}" \
    --step "${SQW_STEP}" | tee "${SQW_METRICS_PATH}"

python - <<PY | tee "${VELOCITY_METRICS_PATH}"
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_sqw_comparison import crystal_frames_to_flat_positions

ase_path = Path("${ASE_OUTPUT_PATH}")
data_path = Path("${DATA_PATH}")
out_path = Path("${VELOCITY_PLOT_PATH}")
dt_ps = float("${DT_PS}")
window = int("${VELOCITY_WINDOW}")
bins = int("${VELOCITY_BINS}")

ase = np.load(ase_path)
data = np.load(data_path)
predicted_velocities = np.asarray(ase["velocities_ang_per_ps"], dtype=np.float64)[1:]
start = int(ase["initial_frames"][-1])
available_reference_frames = max(0, int(data["displacements"].shape[0]) - start)
available_predicted_frames = min(
    int(ase["positions"].shape[0]),
    int(ase["velocities_ang_per_ps"].shape[0]),
)
frames = min(available_predicted_frames, available_reference_frames)
if frames < 2:
    raise ValueError(
        f"Not enough overlapping frames for velocity comparison: "
        f"predicted_available={available_predicted_frames}, "
        f"reference_available={available_reference_frames}, start={start}"
    )
if frames < available_predicted_frames:
    print(
        f"WARNING: trimming velocity comparison from {available_predicted_frames} to {frames} frames "
        f"because reference data ends at frame {int(data['displacements'].shape[0]) - 1}."
    )
predicted_velocities = predicted_velocities[:frames - 1]
reference_displacements = np.asarray(data["displacements"][start:start + frames], dtype=np.float32)
reference_flat = crystal_frames_to_flat_positions(
    reference_displacements,
    data["reference_positions"],
    data["atom_order"],
)
reference_positions = reference_flat.reshape(frames, -1, 3).astype(np.float64)
reference_velocities = np.diff(reference_positions, axis=0) / dt_ps

sets = [
    ("all frames", predicted_velocities, reference_velocities),
    (f"first {window} intervals", predicted_velocities[:window], reference_velocities[:window]),
    (f"last {window} intervals", predicted_velocities[-window:], reference_velocities[-window:]),
]

def components(values):
    return np.asarray(values, dtype=np.float64).reshape(-1)

def speeds(values):
    return np.linalg.norm(np.asarray(values, dtype=np.float64).reshape(-1, 3), axis=1)

def component_limit(*arrays):
    values = np.concatenate([np.abs(components(array)) for array in arrays])
    limit = float(np.nanpercentile(values, 99.5))
    return limit if np.isfinite(limit) and limit > 0 else 1.0

def speed_limit(*arrays):
    values = np.concatenate([speeds(array) for array in arrays])
    limit = float(np.nanpercentile(values, 99.5))
    return limit if np.isfinite(limit) and limit > 0 else 1.0

fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
for column, (label, predicted, reference) in enumerate(sets):
    limit = component_limit(predicted, reference)
    ax = axes[0, column]
    ax.hist(
        components(reference),
        bins=bins,
        range=(-limit, limit),
        density=True,
        alpha=0.48,
        label="reference",
        color="tab:blue",
    )
    ax.hist(
        components(predicted),
        bins=bins,
        range=(-limit, limit),
        density=True,
        alpha=0.48,
        label="ASE inference",
        color="tab:orange",
    )
    ax.set_title(f"v components, {label}")
    ax.set_xlabel("v component, A/ps")
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)
    if column == 0:
        ax.legend(fontsize=9)

    limit = speed_limit(predicted, reference)
    ax = axes[1, column]
    ax.hist(
        speeds(reference),
        bins=bins,
        range=(0, limit),
        density=True,
        alpha=0.48,
        label="reference",
        color="tab:blue",
    )
    ax.hist(
        speeds(predicted),
        bins=bins,
        range=(0, limit),
        density=True,
        alpha=0.48,
        label="ASE inference",
        color="tab:orange",
    )
    ax.set_title(f"|v|, {label}")
    ax.set_xlabel("|v|, A/ps")
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)

fig.suptitle("ASE chained NVT velocity histograms vs data1055 reference", fontsize=15)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=180)
plt.close(fig)

print(f"Saved {out_path}")
print("window\tsource\tspeed_mean\tspeed_std\tspeed_p95\tcomponent_std")
for label, predicted, reference in sets:
    for source, values in [("ASE", predicted), ("reference", reference)]:
        speed = speeds(values)
        component = components(values)
        print(
            f"{label}\t{source}\t{speed.mean():.6g}\t{speed.std():.6g}\t"
            f"{np.percentile(speed, 95):.6g}\t{component.std():.6g}"
        )
PY

echo "ASE_OUTPUT_PATH=${ASE_OUTPUT_PATH}"
echo "SQW_PLOT_PATH=${SQW_PLOT_PATH}"
echo "SQW_METRICS_PATH=${SQW_METRICS_PATH}"
echo "VELOCITY_PLOT_PATH=${VELOCITY_PLOT_PATH}"
echo "VELOCITY_METRICS_PATH=${VELOCITY_METRICS_PATH}"
