"""Run NVT ASE dynamics with the Cu RNN acceleration calculator."""

import argparse
import sys
from pathlib import Path

import numpy as np

from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.md.bussi import Bussi

from ase_copper_calculator import CopperFieldRNNCalculator


DT_PS = 0.02
TEMPERATURE_K = 300.0
TAUT_FS = 200.0


def parse_args():
    """Parse command-line options for ASE NVT inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Saved RNN acceleration model.")
    parser.add_argument("--data-path", required=True, help="Prepared crystal .npz dataset.")
    parser.add_argument("--output-npz", required=True, help="Output trajectory .npz path.")
    parser.add_argument("--trajectory-path", default=None, help="Optional ASE .traj output path.")
    parser.add_argument("--log-path", default="-", help="ASE MD log path, or '-' for stdout.")
    parser.add_argument("--steps", type=int, required=True, help="Number of ASE MD steps.")
    parser.add_argument(
        "--initial-frames",
        type=int,
        nargs=3,
        required=True,
        metavar=("FRAME0", "FRAME1", "FRAME2"),
        help="Three consecutive or chosen frames from data-path used as initial history.",
    )
    parser.add_argument("--temperature-k", type=float, default=TEMPERATURE_K)
    parser.add_argument("--taut-fs", type=float, default=TAUT_FS, help="Bussi thermostat coupling time in fs.")
    parser.add_argument("--dt-ps", type=float, default=DT_PS, help="Model/MD time step in ps.")
    parser.add_argument("--patch-shape", type=int, nargs=3, default=(3, 3, 3))
    parser.add_argument("--patch-batch-size", type=int, default=250)
    parser.add_argument(
        "--periodic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use periodic centered patches in the RNN calculator.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--record-interval", type=int, default=1)
    parser.add_argument(
        "--rescale-initial-temperature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Scale frame-derived initial velocities to temperature-k and adjust the "
            "history frame differences by the same factor."
        ),
    )
    parser.add_argument(
        "--fixcm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compatibility option kept for old commands; ASE Bussi does not use fixcm.",
    )
    parser.add_argument(
        "--q-zero-mode",
        choices=[
            "none",
            "initial",
            "zero",
            "constant_velocity",
            "initial-every-step",
            "zero-every-step",
            "constant-velocity-every-step",
        ],
        default="initial",
        help=(
            "Control the spatial q=0 displacement mode. "
            "'initial' corrects only the initial history and COM velocity; "
            "'zero' removes the initial q=0 displacement and velocity; "
            "'constant_velocity' preserves the initial q=0 velocity; "
            "'*-every-step' variants additionally project q=0 after every MD step."
        ),
    )
    parser.add_argument(
        "--low-q-correction-mode",
        choices=["none", "reference"],
        default="none",
        help=(
            "Optional Fourier-space low-q acceleration correction. 'reference' estimates "
            "k_eff(q) from data-path displacements and blends selected modes toward -k_eff(q) u(q)."
        ),
    )
    parser.add_argument(
        "--low-q-correction-max-q",
        type=float,
        default=0.0,
        help="Largest |q| in 1/A corrected by the low-q mode. Zero disables the correction.",
    )
    parser.add_argument(
        "--low-q-correction-blend",
        type=float,
        default=1.0,
        help="Blend from model acceleration to elastic low-q acceleration: 0=model, 1=elastic.",
    )
    parser.add_argument(
        "--low-q-correction-exclude-q-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude the spatial q=0 Fourier mode from the low-q correction.",
    )
    parser.add_argument(
        "--low-q-reference-max-frames",
        type=int,
        default=None,
        help="Optional frame cap when estimating reference low-q stiffness.",
    )
    parser.add_argument(
        "--low-q-stiffness-epsilon",
        type=float,
        default=1e-30,
        help="Numerical denominator floor for reference low-q stiffness estimation.",
    )
    parser.add_argument(
        "--acceleration-scale",
        type=float,
        default=1.0,
        help="Global multiplier applied to calculator accelerations before converting them to ASE forces.",
    )
    parser.add_argument(
        "--curl-correction-mode",
        choices=["none", "local-linear"],
        default="none",
        help="Optional local Jacobian-vorticity correction applied to model accelerations.",
    )
    parser.add_argument(
        "--curl-correction-eta",
        type=float,
        default=0.0,
        help="Strength of the local-linear curl correction.",
    )
    parser.add_argument(
        "--curl-correction-interval",
        type=int,
        default=10,
        help="Recompute local antisymmetric Jacobians once every N force calls.",
    )
    parser.add_argument(
        "--curl-correction-batch-size",
        type=int,
        default=32,
        help="Patch batch size used when estimating curl-correction Jacobians.",
    )
    parser.add_argument(
        "--curl-correction-epsilon",
        type=float,
        default=1e-12,
        help="Numerical floor reserved for curl-correction diagnostics.",
    )
    parser.add_argument(
        "--history-damping-mode",
        choices=["none", "local-positive"],
        default="none",
        help="Optional correction that removes the positive local d a / d v response.",
    )
    parser.add_argument("--history-damping-eta", type=float, default=0.0, help="History damping strength.")
    parser.add_argument(
        "--history-damping-interval",
        type=int,
        default=10,
        help="Recompute positive velocity-response matrices once every N force calls.",
    )
    parser.add_argument(
        "--history-damping-batch-size",
        type=int,
        default=32,
        help="Patch batch size used when estimating history damping matrices.",
    )
    parser.add_argument(
        "--history-damping-adaptive-gain",
        type=float,
        default=0.0,
        help=(
            "Adaptive eta gain applied to positive normalized power error. "
            "Zero disables adaptive eta updates."
        ),
    )
    parser.add_argument(
        "--history-damping-adaptive-cooling-gain",
        type=float,
        default=None,
        help=(
            "Adaptive eta gain applied to negative normalized power error. "
            "Defaults to history-damping-adaptive-gain for symmetric updates."
        ),
    )
    parser.add_argument(
        "--history-damping-adaptive-interval",
        type=int,
        default=100,
        help="Update adaptive history-damping eta once every N force calls.",
    )
    parser.add_argument(
        "--history-damping-adaptive-min-eta",
        type=float,
        default=0.0,
        help="Lower clamp for adaptive history-damping eta.",
    )
    parser.add_argument(
        "--history-damping-adaptive-max-eta",
        type=float,
        default=None,
        help="Optional upper clamp for adaptive history-damping eta.",
    )
    parser.add_argument(
        "--history-damping-adaptive-target-power",
        type=float,
        default=0.0,
        help="Target normalized mean a*v power for adaptive eta.",
    )
    parser.add_argument(
        "--history-damping-adaptive-ema",
        type=float,
        default=0.05,
        help="EMA factor for normalized power before adaptive eta updates.",
    )
    parser.add_argument(
        "--history-damping-adaptive-epsilon",
        type=float,
        default=1e-30,
        help="Denominator floor for normalized adaptive power.",
    )
    parser.add_argument(
        "--temperature-eta-adaptive-mode",
        choices=["none", "log"],
        default="none",
        help="Adapt history-damping eta from the smoothed ASE temperature.",
    )
    parser.add_argument(
        "--temperature-eta-adaptive-gain",
        type=float,
        default=0.0,
        help="Eta change per log(T_ema / temperature-k) update.",
    )
    parser.add_argument(
        "--temperature-eta-adaptive-interval",
        type=int,
        default=100,
        help="Update temperature-adaptive eta once every N ASE steps.",
    )
    parser.add_argument(
        "--temperature-eta-adaptive-ema",
        type=float,
        default=0.002,
        help="EMA factor for temperature before eta updates.",
    )
    parser.add_argument(
        "--temperature-eta-adaptive-min-eta",
        type=float,
        default=0.0,
        help="Lower clamp for temperature-adaptive eta.",
    )
    parser.add_argument(
        "--temperature-eta-adaptive-max-eta",
        type=float,
        default=None,
        help="Optional upper clamp for temperature-adaptive eta.",
    )
    parser.add_argument(
        "--temperature-eta-adaptive-deadband",
        type=float,
        default=0.0,
        help="Ignore absolute log temperature errors below this value.",
    )
    parser.add_argument(
        "--power-bias-correction-mode",
        choices=["none", "global", "global-positive"],
        default="none",
        help=(
            "Optional global correction that subtracts a component along the current "
            "velocity to control mean acceleration power."
        ),
    )
    parser.add_argument(
        "--power-bias-correction-alpha",
        type=float,
        default=1.0,
        help="Fraction of the measured mean-power bias removed on each force call.",
    )
    parser.add_argument(
        "--power-bias-correction-epsilon",
        type=float,
        default=1e-30,
        help="Denominator floor for global power-bias correction.",
    )
    return parser.parse_args()


def load_crystal_dataset(path):
    """Load arrays needed to construct ASE Atoms and calculator history."""
    data = np.load(path)
    required = ["displacements", "reference_positions", "atom_order"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {missing}")
    return data


def cell_from_dataset(data):
    """Return a 3x3 ASE cell matrix from a prepared dataset."""
    if "cell" in data.files:
        cell = np.asarray(data["cell"], dtype=np.float64)
        if cell.ndim == 3:
            cell = cell[0]
        if cell.shape == (3,):
            return np.diag(cell)
        if cell.shape == (3, 3):
            return cell
    if "box_lengths" in data.files:
        box_lengths = np.asarray(data["box_lengths"], dtype=np.float64)
        if box_lengths.ndim == 2:
            box_lengths = box_lengths[0]
        return np.diag(box_lengths)
    raise ValueError("Dataset must contain either 'cell' or 'box_lengths'")


def crystal_to_flat_values(crystal_values, atom_order):
    """Convert crystal-shaped values to flat ASE atom order."""
    atom_order = np.asarray(atom_order, dtype=np.int64)
    flat = np.empty((atom_order.size, 3), dtype=np.float64)
    flat[atom_order.reshape(-1)] = np.asarray(crystal_values, dtype=np.float64).reshape(atom_order.size, 3)
    return flat


def flat_to_crystal_values(flat_values, atom_order):
    """Convert flat ASE-order values to crystal layout."""
    return np.asarray(flat_values, dtype=np.float64)[np.asarray(atom_order, dtype=np.int64), :]


def frame_positions(data, frame_indices):
    """Return flat ASE-order positions for selected crystal frames."""
    reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
    atom_order = np.asarray(data["atom_order"], dtype=np.int64)
    displacements = np.asarray(data["displacements"][list(frame_indices)], dtype=np.float64)
    frames = []
    for crystal_displacements in displacements:
        frames.append(reference_positions + crystal_to_flat_values(crystal_displacements, atom_order))
    return np.asarray(frames, dtype=np.float64)


def minimum_image_delta(delta, cell, pbc):
    """Apply minimum-image wrapping to position differences."""
    pbc = np.asarray(pbc, dtype=bool)
    if not np.any(pbc):
        return delta
    fractional = np.asarray(delta, dtype=np.float64) @ np.linalg.inv(cell)
    fractional[:, pbc] -= np.round(fractional[:, pbc])
    return fractional @ cell


def spatial_q_zero(crystal_values):
    """Return the per-basis-atom spatial q=0 mode."""
    return np.mean(np.asarray(crystal_values, dtype=np.float64), axis=(0, 1, 2), keepdims=True)


def q_zero_target(step, mode, initial_q_zero, initial_q_zero_delta):
    """Return the requested q=0 displacement mode for one ASE step."""
    mode = normalize_q_zero_mode(mode)
    if mode == "initial":
        return initial_q_zero
    if mode == "constant-velocity":
        return initial_q_zero + float(step) * initial_q_zero_delta
    if mode == "zero":
        return np.zeros_like(initial_q_zero)
    raise ValueError(f"Unsupported q_zero_mode={mode!r}")


def normalize_q_zero_mode(mode):
    """Normalize q-zero mode aliases used by older command lines."""
    return str(mode).replace("_", "-")


def q_zero_initial_mode(mode):
    """Return the initial-history q=0 operation for a selected mode."""
    normalized = normalize_q_zero_mode(mode)
    if normalized.endswith("-every-step"):
        normalized = normalized[: -len("-every-step")]
    if normalized == "constant-velocity":
        return "constant-velocity"
    return normalized


def q_zero_every_step_mode(mode):
    """Return the per-step q=0 operation, or 'none' when only initialization is requested."""
    normalized = normalize_q_zero_mode(mode)
    if not normalized.endswith("-every-step"):
        return "none"
    return normalized[: -len("-every-step")]


def apply_initial_q_zero_mode(data, positions_history, mode):
    """Return initial positions with a consistent spatial q=0 gauge."""
    mode = q_zero_initial_mode(mode)
    if mode == "none":
        return np.asarray(positions_history, dtype=np.float64)

    reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
    atom_order = np.asarray(data["atom_order"], dtype=np.int64)
    history_displacements = []
    for positions in np.asarray(positions_history, dtype=np.float64):
        history_displacements.append(flat_to_crystal_values(positions - reference_positions, atom_order))
    history_displacements = np.asarray(history_displacements, dtype=np.float64)

    latest_q_zero = spatial_q_zero(history_displacements[-1])
    latest_delta = spatial_q_zero(history_displacements[-1]) - spatial_q_zero(history_displacements[-2])
    corrected_frames = []
    latest_index = len(history_displacements) - 1
    for index, displacements in enumerate(history_displacements):
        if mode == "constant-velocity":
            target = latest_q_zero + float(index - latest_index) * latest_delta
        else:
            target = q_zero_target(0, mode, latest_q_zero, latest_delta)
        corrected_displacements = displacements - spatial_q_zero(displacements) + target
        corrected_frames.append(reference_positions + crystal_to_flat_values(corrected_displacements, atom_order))
    return np.asarray(corrected_frames, dtype=np.float64)


class ASEQZeroController:
    """Apply displacement and velocity q=0 corrections to ASE atoms."""

    def __init__(self, data, positions_history, timestep, mode):
        self.mode = q_zero_every_step_mode(mode)
        self.reference_positions = np.asarray(data["reference_positions"], dtype=np.float64)
        self.atom_order = np.asarray(data["atom_order"], dtype=np.int64)
        self.timestep = float(timestep)
        history_displacements = []
        for positions in np.asarray(positions_history, dtype=np.float64):
            history_displacements.append(flat_to_crystal_values(positions - self.reference_positions, self.atom_order))
        history_displacements = np.asarray(history_displacements, dtype=np.float64)
        self.initial_q_zero = spatial_q_zero(history_displacements[-1])
        self.initial_q_zero_delta = spatial_q_zero(history_displacements[-1]) - spatial_q_zero(history_displacements[-2])
        self.target_velocity_q_zero = self._target_velocity_q_zero()

    def _target_velocity_q_zero(self):
        """Return the q=0 velocity target in ASE internal velocity units."""
        if self.mode == "constant-velocity":
            return self.initial_q_zero_delta / self.timestep
        return np.zeros_like(self.initial_q_zero)

    def apply(self, atoms, step):
        """Correct ASE positions, velocities, and calculator history."""
        if self.mode == "none":
            return

        calculator = atoms.calc
        displacements = calculator._positions_to_crystal_displacements(atoms.get_positions(), atoms=atoms).astype(
            np.float64
        )
        target = q_zero_target(step, self.mode, self.initial_q_zero, self.initial_q_zero_delta)
        corrected_displacements = displacements - spatial_q_zero(displacements) + target
        corrected_positions = self.reference_positions + crystal_to_flat_values(corrected_displacements, self.atom_order)
        atoms.set_positions(corrected_positions)

        velocities = atoms.get_velocities()
        if velocities is not None:
            velocity_crystal = flat_to_crystal_values(velocities, self.atom_order)
            corrected_velocity = velocity_crystal - spatial_q_zero(velocity_crystal) + self.target_velocity_q_zero
            atoms.set_velocities(crystal_to_flat_values(corrected_velocity, self.atom_order))

        calculator.replace_current_history_positions(atoms.get_positions(), atoms=atoms)
        calculator.results.clear()


def kinetic_temperature_from_velocities(velocities, masses):
    """Return the classical kinetic temperature for velocities in ASE units."""
    kinetic_energy = 0.5 * np.sum(np.asarray(masses, dtype=np.float64)[:, None] * velocities**2)
    dof = max(1, 3 * len(masses))
    return 2.0 * kinetic_energy / (dof * units.kB)


def initialize_history_and_velocities(positions_history, cell, dt_ps, target_temperature_k, rescale_temperature):
    """Return possibly rescaled position history and ASE velocities."""
    timestep = dt_ps * 1000.0 * units.fs
    pbc = np.ones(3, dtype=bool)
    last_delta = minimum_image_delta(positions_history[-1] - positions_history[-2], cell, pbc)
    velocities = last_delta / timestep

    if not rescale_temperature:
        return positions_history, velocities, 1.0

    masses = np.full(len(positions_history[-1]), 63.546, dtype=np.float64)
    current_temperature = kinetic_temperature_from_velocities(velocities, masses)
    if current_temperature <= 0:
        raise ValueError("Cannot rescale zero initial velocity to the requested temperature")
    scale = float(np.sqrt(target_temperature_k / current_temperature))

    scaled_history = positions_history.copy()
    for index in range(len(scaled_history) - 2, -1, -1):
        delta = minimum_image_delta(positions_history[index + 1] - positions_history[index], cell, pbc)
        scaled_history[index] = scaled_history[index + 1] - scale * delta
    return scaled_history, velocities * scale, scale


def write_initial_summary(atoms, args, initial_velocity_scale):
    """Print a compact summary before running MD."""
    print("ASE NVT Cu RNN run")
    print(f"model_path = {args.model_path}")
    print(f"data_path = {args.data_path}")
    print(f"steps = {args.steps}")
    print(f"initial_frames = {tuple(args.initial_frames)}")
    print(f"temperature_k = {args.temperature_k}")
    print(f"taut_fs = {args.taut_fs}")
    print(f"dt_ps = {args.dt_ps}")
    print(f"q_zero_mode = {args.q_zero_mode}")
    print(f"low_q_correction_mode = {args.low_q_correction_mode}")
    print(f"low_q_correction_max_q = {args.low_q_correction_max_q}")
    print(f"low_q_correction_blend = {args.low_q_correction_blend}")
    print(f"low_q_correction_exclude_q_zero = {args.low_q_correction_exclude_q_zero}")
    print(f"low_q_reference_max_frames = {args.low_q_reference_max_frames}")
    print(f"acceleration_scale = {args.acceleration_scale}")
    print(f"curl_correction_mode = {args.curl_correction_mode}")
    print(f"curl_correction_eta = {args.curl_correction_eta}")
    print(f"curl_correction_interval = {args.curl_correction_interval}")
    print(f"curl_correction_batch_size = {args.curl_correction_batch_size}")
    print(f"history_damping_mode = {args.history_damping_mode}")
    print(f"history_damping_eta = {args.history_damping_eta}")
    print(f"history_damping_interval = {args.history_damping_interval}")
    print(f"history_damping_batch_size = {args.history_damping_batch_size}")
    print(f"history_damping_adaptive_gain = {args.history_damping_adaptive_gain}")
    print(f"history_damping_adaptive_cooling_gain = {args.history_damping_adaptive_cooling_gain}")
    print(f"history_damping_adaptive_interval = {args.history_damping_adaptive_interval}")
    print(f"history_damping_adaptive_min_eta = {args.history_damping_adaptive_min_eta}")
    print(f"history_damping_adaptive_max_eta = {args.history_damping_adaptive_max_eta}")
    print(f"history_damping_adaptive_target_power = {args.history_damping_adaptive_target_power}")
    print(f"history_damping_adaptive_ema = {args.history_damping_adaptive_ema}")
    print(f"temperature_eta_adaptive_mode = {args.temperature_eta_adaptive_mode}")
    print(f"temperature_eta_adaptive_gain = {args.temperature_eta_adaptive_gain}")
    print(f"temperature_eta_adaptive_interval = {args.temperature_eta_adaptive_interval}")
    print(f"temperature_eta_adaptive_ema = {args.temperature_eta_adaptive_ema}")
    print(f"temperature_eta_adaptive_min_eta = {args.temperature_eta_adaptive_min_eta}")
    print(f"temperature_eta_adaptive_max_eta = {args.temperature_eta_adaptive_max_eta}")
    print(f"temperature_eta_adaptive_deadband = {args.temperature_eta_adaptive_deadband}")
    print(f"power_bias_correction_mode = {args.power_bias_correction_mode}")
    print(f"power_bias_correction_alpha = {args.power_bias_correction_alpha}")
    print(f"power_bias_correction_epsilon = {args.power_bias_correction_epsilon}")
    if getattr(atoms.calc, "low_q_mask", None) is not None:
        print(f"low_q_mode_count = {atoms.calc.low_q_mode_count}")
    print(f"initial_temperature_k = {atoms.get_temperature():.8g}")
    print(f"initial_velocity_scale = {initial_velocity_scale:.8g}")


def main():
    """Run ASE NVT dynamics and save a compact trajectory npz."""
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.temperature_k <= 0:
        raise ValueError("temperature-k must be positive")
    if args.taut_fs <= 0:
        raise ValueError("taut-fs must be positive")
    if args.dt_ps <= 0:
        raise ValueError("dt-ps must be positive")
    if args.record_interval <= 0:
        raise ValueError("record-interval must be positive")
    if args.acceleration_scale < 0:
        raise ValueError("acceleration-scale must be non-negative")
    if args.curl_correction_eta < 0:
        raise ValueError("curl-correction-eta must be non-negative")
    if args.curl_correction_interval <= 0:
        raise ValueError("curl-correction-interval must be positive")
    if args.curl_correction_batch_size <= 0:
        raise ValueError("curl-correction-batch-size must be positive")
    if args.curl_correction_epsilon <= 0:
        raise ValueError("curl-correction-epsilon must be positive")
    if args.history_damping_eta < 0:
        raise ValueError("history-damping-eta must be non-negative")
    if args.history_damping_interval <= 0:
        raise ValueError("history-damping-interval must be positive")
    if args.history_damping_batch_size <= 0:
        raise ValueError("history-damping-batch-size must be positive")
    if args.history_damping_adaptive_gain < 0:
        raise ValueError("history-damping-adaptive-gain must be non-negative")
    if args.history_damping_adaptive_cooling_gain is not None and args.history_damping_adaptive_cooling_gain < 0:
        raise ValueError("history-damping-adaptive-cooling-gain must be non-negative")
    if args.history_damping_adaptive_interval <= 0:
        raise ValueError("history-damping-adaptive-interval must be positive")
    if args.history_damping_adaptive_min_eta < 0:
        raise ValueError("history-damping-adaptive-min-eta must be non-negative")
    if (
        args.history_damping_adaptive_max_eta is not None
        and args.history_damping_adaptive_max_eta < args.history_damping_adaptive_min_eta
    ):
        raise ValueError("history-damping-adaptive-max-eta must be >= history-damping-adaptive-min-eta")
    if not 0 < args.history_damping_adaptive_ema <= 1:
        raise ValueError("history-damping-adaptive-ema must be in (0, 1]")
    if args.history_damping_adaptive_epsilon <= 0:
        raise ValueError("history-damping-adaptive-epsilon must be positive")
    if args.temperature_eta_adaptive_gain < 0:
        raise ValueError("temperature-eta-adaptive-gain must be non-negative")
    if args.temperature_eta_adaptive_interval <= 0:
        raise ValueError("temperature-eta-adaptive-interval must be positive")
    if not 0 < args.temperature_eta_adaptive_ema <= 1:
        raise ValueError("temperature-eta-adaptive-ema must be in (0, 1]")
    if args.temperature_eta_adaptive_min_eta < 0:
        raise ValueError("temperature-eta-adaptive-min-eta must be non-negative")
    if (
        args.temperature_eta_adaptive_max_eta is not None
        and args.temperature_eta_adaptive_max_eta < args.temperature_eta_adaptive_min_eta
    ):
        raise ValueError("temperature-eta-adaptive-max-eta must be >= temperature-eta-adaptive-min-eta")
    if args.temperature_eta_adaptive_deadband < 0:
        raise ValueError("temperature-eta-adaptive-deadband must be non-negative")
    if args.power_bias_correction_alpha < 0:
        raise ValueError("power-bias-correction-alpha must be non-negative")
    if args.power_bias_correction_epsilon <= 0:
        raise ValueError("power-bias-correction-epsilon must be positive")

    data = load_crystal_dataset(args.data_path)
    frame_count = int(data["displacements"].shape[0])
    if any(frame < 0 or frame >= frame_count for frame in args.initial_frames):
        raise ValueError(f"initial-frames must be in [0, {frame_count})")

    cell = cell_from_dataset(data)
    positions_history = apply_initial_q_zero_mode(
        data=data,
        positions_history=frame_positions(data, args.initial_frames),
        mode=args.q_zero_mode,
    )
    positions_history, velocities, initial_velocity_scale = initialize_history_and_velocities(
        positions_history=positions_history,
        cell=cell,
        dt_ps=args.dt_ps,
        target_temperature_k=args.temperature_k,
        rescale_temperature=args.rescale_initial_temperature,
    )

    atoms = Atoms(
        symbols=["Cu"] * positions_history.shape[1],
        positions=positions_history[-1],
        cell=cell,
        pbc=True,
    )
    atoms.set_velocities(velocities)
    atoms.calc = CopperFieldRNNCalculator(
        model_path=args.model_path,
        data_path=args.data_path,
        dt_ps=args.dt_ps,
        history_positions=positions_history,
        patch_shape=tuple(args.patch_shape),
        periodic=args.periodic,
        patch_batch_size=args.patch_batch_size,
        device=args.device,
        low_q_correction_mode=args.low_q_correction_mode,
        low_q_correction_max_q=args.low_q_correction_max_q,
        low_q_correction_blend=args.low_q_correction_blend,
        low_q_correction_exclude_q_zero=args.low_q_correction_exclude_q_zero,
        low_q_reference_max_frames=args.low_q_reference_max_frames,
        low_q_stiffness_epsilon=args.low_q_stiffness_epsilon,
        acceleration_scale=args.acceleration_scale,
        curl_correction_mode=args.curl_correction_mode,
        curl_correction_eta=args.curl_correction_eta,
        curl_correction_interval=args.curl_correction_interval,
        curl_correction_batch_size=args.curl_correction_batch_size,
        curl_correction_epsilon=args.curl_correction_epsilon,
        history_damping_mode=args.history_damping_mode,
        history_damping_eta=args.history_damping_eta,
        history_damping_interval=args.history_damping_interval,
        history_damping_batch_size=args.history_damping_batch_size,
        history_damping_adaptive_gain=args.history_damping_adaptive_gain,
        history_damping_adaptive_cooling_gain=args.history_damping_adaptive_cooling_gain,
        history_damping_adaptive_interval=args.history_damping_adaptive_interval,
        history_damping_adaptive_min_eta=args.history_damping_adaptive_min_eta,
        history_damping_adaptive_max_eta=args.history_damping_adaptive_max_eta,
        history_damping_adaptive_target_power=args.history_damping_adaptive_target_power,
        history_damping_adaptive_ema=args.history_damping_adaptive_ema,
        history_damping_adaptive_epsilon=args.history_damping_adaptive_epsilon,
        power_bias_correction_mode=args.power_bias_correction_mode,
        power_bias_correction_alpha=args.power_bias_correction_alpha,
        power_bias_correction_epsilon=args.power_bias_correction_epsilon,
    )

    timestep = args.dt_ps * 1000.0 * units.fs
    q_zero_controller = ASEQZeroController(
        data=data,
        positions_history=positions_history,
        timestep=timestep,
        mode=args.q_zero_mode,
    )
    q_zero_controller.apply(atoms, step=0)

    write_initial_summary(atoms, args, initial_velocity_scale)

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    if args.trajectory_path:
        trajectory_path = Path(args.trajectory_path)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory = Trajectory(str(trajectory_path), "w", atoms)
    else:
        trajectory = None

    dynamics = Bussi(
        atoms,
        timestep=timestep,
        temperature_K=args.temperature_k,
        taut=args.taut_fs * units.fs,
        logfile=sys.stdout if args.log_path == "-" else args.log_path,
    )

    recorded_steps = []
    recorded_positions = []
    recorded_velocities = []
    recorded_forces = []
    recorded_temperature = []
    recorded_kinetic_energy = []
    recorded_potential_energy = []
    recorded_history_damping_eta = []
    recorded_history_damping_power_ema = []
    recorded_temperature_eta_ema = []
    temperature_eta_ema = float(atoms.get_temperature())
    temperature_eta_diagnostics = []

    def record(step):
        if recorded_steps and recorded_steps[-1] == step:
            return
        recorded_steps.append(step)
        recorded_positions.append(atoms.get_positions().copy())
        recorded_velocities.append(atoms.get_velocities().copy() * (1000.0 * units.fs))
        recorded_forces.append(atoms.get_forces().copy())
        recorded_temperature.append(float(atoms.get_temperature()))
        recorded_kinetic_energy.append(float(atoms.get_kinetic_energy()))
        recorded_potential_energy.append(float(atoms.get_potential_energy()))
        recorded_history_damping_eta.append(float(getattr(atoms.calc, "history_damping_eta", np.nan)))
        power_ema = getattr(atoms.calc, "_history_damping_power_ema", None)
        recorded_history_damping_power_ema.append(np.nan if power_ema is None else float(power_ema))
        recorded_temperature_eta_ema.append(float(temperature_eta_ema))
        if trajectory is not None:
            trajectory.write(atoms)

    record(0)

    def attached_q_zero_control():
        q_zero_controller.apply(atoms, dynamics.nsteps)

    def attached_temperature_eta_control():
        nonlocal temperature_eta_ema
        if args.temperature_eta_adaptive_mode == "none" or args.temperature_eta_adaptive_gain <= 0:
            return
        temperature = float(atoms.get_temperature())
        alpha = float(args.temperature_eta_adaptive_ema)
        temperature_eta_ema = (1.0 - alpha) * temperature_eta_ema + alpha * temperature
        step = int(dynamics.nsteps)
        if step <= 0 or step % args.temperature_eta_adaptive_interval != 0:
            return

        old_eta = float(getattr(atoms.calc, "history_damping_eta", np.nan))
        log_error = float(np.log(max(temperature_eta_ema, 1e-30) / args.temperature_k))
        effective_error = 0.0 if abs(log_error) < args.temperature_eta_adaptive_deadband else log_error
        eta = old_eta + args.temperature_eta_adaptive_gain * effective_error
        eta = max(float(args.temperature_eta_adaptive_min_eta), eta)
        if args.temperature_eta_adaptive_max_eta is not None:
            eta = min(float(args.temperature_eta_adaptive_max_eta), eta)
        atoms.calc.history_damping_eta = float(eta)
        temperature_eta_diagnostics.append(
            (
                step,
                old_eta,
                float(eta),
                temperature,
                float(temperature_eta_ema),
                log_error,
                effective_error,
            )
        )

    def attached_record():
        record(dynamics.nsteps)

    dynamics.attach(attached_q_zero_control, interval=1)
    dynamics.attach(attached_temperature_eta_control, interval=1)
    dynamics.attach(attached_record, interval=args.record_interval)
    dynamics.run(args.steps)
    if trajectory is not None:
        trajectory.close()

    np.savez_compressed(
        output_npz,
        steps=np.asarray(recorded_steps, dtype=np.int64),
        positions=np.asarray(recorded_positions, dtype=np.float32),
        velocities_ang_per_ps=np.asarray(recorded_velocities, dtype=np.float32),
        forces_ev_per_ang=np.asarray(recorded_forces, dtype=np.float32),
        temperature_k=np.asarray(recorded_temperature, dtype=np.float32),
        kinetic_energy_ev=np.asarray(recorded_kinetic_energy, dtype=np.float32),
        potential_energy_ev=np.asarray(recorded_potential_energy, dtype=np.float32),
        initial_frames=np.asarray(args.initial_frames, dtype=np.int64),
        initial_history_positions=np.asarray(positions_history, dtype=np.float32),
        initial_velocity_scale=np.asarray(initial_velocity_scale, dtype=np.float32),
        cell=np.asarray(cell, dtype=np.float32),
        dt_ps=np.asarray(args.dt_ps, dtype=np.float32),
        temperature_target_k=np.asarray(args.temperature_k, dtype=np.float32),
        taut_fs=np.asarray(args.taut_fs, dtype=np.float32),
        q_zero_mode=np.asarray(args.q_zero_mode),
        low_q_correction_mode=np.asarray(args.low_q_correction_mode),
        low_q_correction_max_q=np.asarray(args.low_q_correction_max_q, dtype=np.float32),
        low_q_correction_blend=np.asarray(args.low_q_correction_blend, dtype=np.float32),
        low_q_correction_exclude_q_zero=np.asarray(args.low_q_correction_exclude_q_zero),
        low_q_reference_max_frames=np.asarray(
            -1 if args.low_q_reference_max_frames is None else args.low_q_reference_max_frames,
            dtype=np.int64,
        ),
        low_q_stiffness_epsilon=np.asarray(args.low_q_stiffness_epsilon, dtype=np.float32),
        low_q_mode_count=np.asarray(getattr(atoms.calc, "low_q_mode_count", 0), dtype=np.int64),
        acceleration_scale=np.asarray(args.acceleration_scale, dtype=np.float32),
        curl_correction_mode=np.asarray(args.curl_correction_mode),
        curl_correction_eta=np.asarray(args.curl_correction_eta, dtype=np.float32),
        curl_correction_interval=np.asarray(args.curl_correction_interval, dtype=np.int64),
        curl_correction_batch_size=np.asarray(args.curl_correction_batch_size, dtype=np.int64),
        curl_correction_epsilon=np.asarray(args.curl_correction_epsilon, dtype=np.float32),
        history_damping_mode=np.asarray(args.history_damping_mode),
        history_damping_eta=np.asarray(args.history_damping_eta, dtype=np.float32),
        history_damping_interval=np.asarray(args.history_damping_interval, dtype=np.int64),
        history_damping_batch_size=np.asarray(args.history_damping_batch_size, dtype=np.int64),
        history_damping_adaptive_gain=np.asarray(args.history_damping_adaptive_gain, dtype=np.float32),
        history_damping_adaptive_cooling_gain=np.asarray(
            args.history_damping_adaptive_gain
            if args.history_damping_adaptive_cooling_gain is None
            else args.history_damping_adaptive_cooling_gain,
            dtype=np.float32,
        ),
        history_damping_adaptive_interval=np.asarray(args.history_damping_adaptive_interval, dtype=np.int64),
        history_damping_adaptive_min_eta=np.asarray(args.history_damping_adaptive_min_eta, dtype=np.float32),
        history_damping_adaptive_max_eta=np.asarray(
            np.nan if args.history_damping_adaptive_max_eta is None else args.history_damping_adaptive_max_eta,
            dtype=np.float32,
        ),
        history_damping_adaptive_target_power=np.asarray(
            args.history_damping_adaptive_target_power,
            dtype=np.float32,
        ),
        history_damping_adaptive_ema=np.asarray(args.history_damping_adaptive_ema, dtype=np.float32),
        history_damping_adaptive_epsilon=np.asarray(args.history_damping_adaptive_epsilon, dtype=np.float32),
        history_damping_eta_recorded=np.asarray(recorded_history_damping_eta, dtype=np.float32),
        history_damping_power_ema_recorded=np.asarray(recorded_history_damping_power_ema, dtype=np.float32),
        temperature_eta_adaptive_mode=np.asarray(args.temperature_eta_adaptive_mode),
        temperature_eta_adaptive_gain=np.asarray(args.temperature_eta_adaptive_gain, dtype=np.float32),
        temperature_eta_adaptive_interval=np.asarray(args.temperature_eta_adaptive_interval, dtype=np.int64),
        temperature_eta_adaptive_ema=np.asarray(args.temperature_eta_adaptive_ema, dtype=np.float32),
        temperature_eta_adaptive_min_eta=np.asarray(args.temperature_eta_adaptive_min_eta, dtype=np.float32),
        temperature_eta_adaptive_max_eta=np.asarray(
            np.nan if args.temperature_eta_adaptive_max_eta is None else args.temperature_eta_adaptive_max_eta,
            dtype=np.float32,
        ),
        temperature_eta_adaptive_deadband=np.asarray(args.temperature_eta_adaptive_deadband, dtype=np.float32),
        temperature_eta_ema_recorded=np.asarray(recorded_temperature_eta_ema, dtype=np.float32),
        temperature_eta_diagnostics=np.asarray(temperature_eta_diagnostics, dtype=np.float64),
        history_damping_diagnostics=np.asarray(
            getattr(atoms.calc, "history_damping_diagnostics", []),
            dtype=np.float64,
        ),
        history_damping_eta_final=np.asarray(getattr(atoms.calc, "history_damping_eta", np.nan), dtype=np.float32),
        power_bias_correction_mode=np.asarray(args.power_bias_correction_mode),
        power_bias_correction_alpha=np.asarray(args.power_bias_correction_alpha, dtype=np.float32),
        power_bias_correction_epsilon=np.asarray(args.power_bias_correction_epsilon, dtype=np.float32),
        power_bias_correction_diagnostics=np.asarray(
            getattr(atoms.calc, "power_bias_correction_diagnostics", []),
            dtype=np.float64,
        ),
        initial_q_zero=q_zero_controller.initial_q_zero.astype(np.float32),
        initial_q_zero_delta=q_zero_controller.initial_q_zero_delta.astype(np.float32),
        model_path=np.asarray(str(args.model_path)),
        data_path=np.asarray(str(args.data_path)),
    )
    print(f"Saved {output_npz}")
    if args.trajectory_path:
        print(f"Saved {args.trajectory_path}")


if __name__ == "__main__":
    main()
