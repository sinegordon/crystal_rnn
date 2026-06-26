"""ASE calculator wrapper for copper RNN acceleration models.

Most current RNN crystal architectures are dynamical models: they predict a
discrete acceleration from a short displacement history.  This calculator
therefore keeps an internal history of ASE positions and returns effective
forces

    F = m * (u_next - 2 u_current + u_previous) / dt**2

in ASE units.  Energy-based pair models additionally report the learned scalar
pair potential converted to eV; other backends still report a dummy zero.
"""

from __future__ import annotations

import numpy as np
import torch

from infer_field_rnn_centered_acceleration import (
    build_centers,
    extract_patch_batch,
    predict_center_accelerations,
    resolve_device,
    shape3,
)

try:
    from ase.calculators.calculator import Calculator, all_changes
except ImportError:  # pragma: no cover - exercised only when ASE is absent.
    Calculator = object
    all_changes = ("positions", "numbers", "cell", "pbc")


CU_ATOMIC_NUMBER = 29
CU_MASS_AMU = 63.546
AMU_ANGSTROM_PER_PS2_TO_EV_PER_ANGSTROM = 1.0364269656262175e-4


def _as_cell_matrix(cell):
    """Return an ASE-like cell as a 3x3 matrix with row cell vectors."""
    matrix = np.asarray(cell, dtype=np.float64)
    if matrix.shape == (3,):
        matrix = np.diag(matrix)
    if matrix.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3) or contain three orthorhombic lengths")
    return matrix


def _reference_cell_from_npz(data):
    """Return a reference cell stored in a prepared crystal dataset, if present."""
    if "cell" in data.files:
        cell = np.asarray(data["cell"], dtype=np.float64)
        if cell.ndim == 3:
            cell = cell[0]
        return _as_cell_matrix(cell)
    if "box_lengths" in data.files:
        box_lengths = np.asarray(data["box_lengths"], dtype=np.float64)
        if box_lengths.ndim == 2:
            box_lengths = box_lengths[0]
        return _as_cell_matrix(box_lengths)
    return None


def _q_abs_grid(shape, cell):
    """Return |q| values in FFT index order for a rectangular crystal."""
    lengths = np.linalg.norm(np.asarray(cell, dtype=np.float64), axis=1)
    axes = [2.0 * np.pi * np.fft.fftfreq(int(n), d=float(length) / int(n)) for n, length in zip(shape, lengths)]
    qx, qy, qz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.sqrt(qx**2 + qy**2 + qz**2)


def _reference_effective_stiffness(data, max_frames=None, epsilon=1e-30):
    """Estimate k_eff(q) = -Re<a_q u_q*> / <|u_q|^2> from reference displacements."""
    if "displacements" not in data.files:
        raise ValueError("reference low-q correction requires displacements in data_path")
    displacements = np.asarray(data["displacements"], dtype=np.float64)
    if max_frames is not None:
        displacements = displacements[: int(max_frames)]
    if displacements.shape[0] < 3:
        raise ValueError("at least three displacement frames are required for low-q stiffness")

    centered = displacements[1:-1]
    acceleration = displacements[2:] - 2.0 * centered + displacements[:-2]
    cell_count = int(np.prod(displacements.shape[1:4]))
    scale = np.sqrt(cell_count)
    u_spectrum = np.fft.fftn(centered, axes=(1, 2, 3)) / scale
    a_spectrum = np.fft.fftn(acceleration, axes=(1, 2, 3)) / scale
    numerator = np.sum(-np.real(a_spectrum * np.conj(u_spectrum)), axis=(0, 4, 5))
    denominator = np.sum(np.abs(u_spectrum) ** 2, axis=(0, 4, 5))
    stiffness = numerator / (denominator + float(epsilon))
    return np.maximum(stiffness, 0.0).astype(np.float64)


def _load_reference_data(path):
    """Load only the arrays needed by the ASE calculator."""
    data = np.load(path)
    required = ["reference_positions", "atom_order"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {missing}")
    return data


class CopperFieldRNNCalculator(Calculator):
    """Stateful ASE calculator backed by a Cu RNN acceleration model.

    Args:
        model_path: Saved acceleration model.  Supported backends are
            ``CrystalFieldRNNNet`` with ``target_mode='acceleration'`` and
            ``CrystalEdgeRNNNet`` and the experimental pair-force / pair-energy
            models.
        data_path: Prepared ``.npz`` dataset containing ``reference_positions``
            and ``atom_order``.  ``box_lengths`` or ``cell`` is used when
            present to define the reference periodic cell.
        dt_ps: Time step between model trajectory frames, in picoseconds.
        sequence_length: Number of history frames expected by the model.  If
            omitted, ``sequence_length`` from the dataset is used when present,
            otherwise the current project default ``3`` is used.
        history_positions: Optional initial position history with shape
            ``(sequence_length, atoms, 3)`` in Angstrom and in ASE atom order.
        patch_shape: Centered local patch shape used for inference.
        periodic: Whether local patches wrap around crystal boundaries.
        patch_batch_size: Number of centered patches evaluated per torch batch.
        device: Torch device: ``auto``, ``cpu``, ``cuda``, etc.
        use_atoms_masses: If true, use masses from the ASE ``Atoms`` object;
            otherwise use the standard copper mass for every atom.
        enforce_cell: If true, reject calls with a cell different from the
            reference cell.
        history_tolerance: Position-displacement tolerance used to detect
            repeated ASE force calls for the same configuration.
        low_q_correction_mode: Optional long-wavelength correction. ``none``
            leaves model forces unchanged. ``reference`` estimates an effective
            harmonic stiffness from ``data_path`` and blends selected low-q
            model accelerations toward ``-k_ref(q) u(q)``.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_path,
        data_path,
        dt_ps=0.02,
        sequence_length=None,
        history_positions=None,
        patch_shape=(3, 3, 3),
        periodic=True,
        patch_batch_size=250,
        device="auto",
        use_atoms_masses=True,
        enforce_cell=True,
        history_tolerance=1e-8,
        low_q_correction_mode="none",
        low_q_correction_max_q=0.0,
        low_q_correction_blend=1.0,
        low_q_correction_exclude_q_zero=True,
        low_q_reference_max_frames=None,
        low_q_stiffness_epsilon=1e-30,
        acceleration_scale=1.0,
        curl_correction_mode="none",
        curl_correction_eta=0.0,
        curl_correction_interval=10,
        curl_correction_batch_size=32,
        curl_correction_epsilon=1e-12,
        history_damping_mode="none",
        history_damping_eta=0.0,
        history_damping_interval=10,
        history_damping_batch_size=32,
        history_damping_adaptive_gain=0.0,
        history_damping_adaptive_cooling_gain=None,
        history_damping_adaptive_interval=100,
        history_damping_adaptive_min_eta=0.0,
        history_damping_adaptive_max_eta=None,
        history_damping_adaptive_target_power=0.0,
        history_damping_adaptive_ema=0.05,
        history_damping_adaptive_epsilon=1e-30,
        power_bias_correction_mode="none",
        power_bias_correction_alpha=1.0,
        power_bias_correction_epsilon=1e-30,
        **kwargs,
    ):
        if Calculator is object:
            raise ImportError("ASE is required to instantiate CopperFieldRNNCalculator")
        super().__init__(**kwargs)

        if dt_ps <= 0:
            raise ValueError("dt_ps must be positive")
        if patch_batch_size <= 0:
            raise ValueError("patch_batch_size must be positive")

        self.model_path = str(model_path)
        self.data_path = str(data_path)
        self.dt_ps = float(dt_ps)
        self.patch_shape = shape3("patch_shape", patch_shape)
        self.periodic = bool(periodic)
        self.patch_batch_size = int(patch_batch_size)
        self.device = resolve_device(device)
        self.use_atoms_masses = bool(use_atoms_masses)
        self.enforce_cell = bool(enforce_cell)
        self.history_tolerance = float(history_tolerance)
        self.low_q_correction_mode = str(low_q_correction_mode)
        self.low_q_correction_max_q = float(low_q_correction_max_q)
        self.low_q_correction_blend = float(low_q_correction_blend)
        self.low_q_correction_exclude_q_zero = bool(low_q_correction_exclude_q_zero)
        self.low_q_reference_max_frames = (
            None if low_q_reference_max_frames is None else int(low_q_reference_max_frames)
        )
        self.low_q_stiffness_epsilon = float(low_q_stiffness_epsilon)
        self.acceleration_scale = float(acceleration_scale)
        self.curl_correction_mode = str(curl_correction_mode)
        self.curl_correction_eta = float(curl_correction_eta)
        self.curl_correction_interval = int(curl_correction_interval)
        self.curl_correction_batch_size = int(curl_correction_batch_size)
        self.curl_correction_epsilon = float(curl_correction_epsilon)
        self.history_damping_mode = str(history_damping_mode)
        self.history_damping_eta = float(history_damping_eta)
        self.history_damping_interval = int(history_damping_interval)
        self.history_damping_batch_size = int(history_damping_batch_size)
        self.history_damping_adaptive_gain = float(history_damping_adaptive_gain)
        self.history_damping_adaptive_cooling_gain = (
            self.history_damping_adaptive_gain
            if history_damping_adaptive_cooling_gain is None
            else float(history_damping_adaptive_cooling_gain)
        )
        self.history_damping_adaptive_interval = int(history_damping_adaptive_interval)
        self.history_damping_adaptive_min_eta = float(history_damping_adaptive_min_eta)
        self.history_damping_adaptive_max_eta = (
            None if history_damping_adaptive_max_eta is None else float(history_damping_adaptive_max_eta)
        )
        self.history_damping_adaptive_target_power = float(history_damping_adaptive_target_power)
        self.history_damping_adaptive_ema = float(history_damping_adaptive_ema)
        self.history_damping_adaptive_epsilon = float(history_damping_adaptive_epsilon)
        self.power_bias_correction_mode = str(power_bias_correction_mode)
        self.power_bias_correction_alpha = float(power_bias_correction_alpha)
        self.power_bias_correction_epsilon = float(power_bias_correction_epsilon)
        if self.low_q_correction_mode not in {"none", "reference"}:
            raise ValueError("low_q_correction_mode must be 'none' or 'reference'")
        if self.curl_correction_mode not in {"none", "local-linear"}:
            raise ValueError("curl_correction_mode must be 'none' or 'local-linear'")
        if self.history_damping_mode not in {"none", "local-positive"}:
            raise ValueError("history_damping_mode must be 'none' or 'local-positive'")
        if self.power_bias_correction_mode not in {"none", "global", "global-positive"}:
            raise ValueError("power_bias_correction_mode must be 'none', 'global', or 'global-positive'")
        if self.low_q_correction_max_q < 0:
            raise ValueError("low_q_correction_max_q must be non-negative")
        if not 0.0 <= self.low_q_correction_blend <= 1.0:
            raise ValueError("low_q_correction_blend must be between 0 and 1")
        if self.low_q_stiffness_epsilon <= 0:
            raise ValueError("low_q_stiffness_epsilon must be positive")
        if self.acceleration_scale < 0:
            raise ValueError("acceleration_scale must be non-negative")
        if self.curl_correction_eta < 0:
            raise ValueError("curl_correction_eta must be non-negative")
        if self.curl_correction_interval <= 0:
            raise ValueError("curl_correction_interval must be positive")
        if self.curl_correction_batch_size <= 0:
            raise ValueError("curl_correction_batch_size must be positive")
        if self.curl_correction_epsilon <= 0:
            raise ValueError("curl_correction_epsilon must be positive")
        if self.history_damping_eta < 0:
            raise ValueError("history_damping_eta must be non-negative")
        if self.history_damping_interval <= 0:
            raise ValueError("history_damping_interval must be positive")
        if self.history_damping_batch_size <= 0:
            raise ValueError("history_damping_batch_size must be positive")
        if self.history_damping_adaptive_gain < 0:
            raise ValueError("history_damping_adaptive_gain must be non-negative")
        if self.history_damping_adaptive_cooling_gain < 0:
            raise ValueError("history_damping_adaptive_cooling_gain must be non-negative")
        if self.history_damping_adaptive_interval <= 0:
            raise ValueError("history_damping_adaptive_interval must be positive")
        if self.history_damping_adaptive_min_eta < 0:
            raise ValueError("history_damping_adaptive_min_eta must be non-negative")
        if (
            self.history_damping_adaptive_max_eta is not None
            and self.history_damping_adaptive_max_eta < self.history_damping_adaptive_min_eta
        ):
            raise ValueError("history_damping_adaptive_max_eta must be >= history_damping_adaptive_min_eta")
        if not 0 < self.history_damping_adaptive_ema <= 1:
            raise ValueError("history_damping_adaptive_ema must be in (0, 1]")
        if self.history_damping_adaptive_epsilon <= 0:
            raise ValueError("history_damping_adaptive_epsilon must be positive")
        if self.power_bias_correction_alpha < 0:
            raise ValueError("power_bias_correction_alpha must be non-negative")
        if self.power_bias_correction_epsilon <= 0:
            raise ValueError("power_bias_correction_epsilon must be positive")

        data = _load_reference_data(data_path)
        self.reference_positions = np.asarray(data["reference_positions"], dtype=np.float32)
        self.atom_order = np.asarray(data["atom_order"], dtype=np.int64)
        self.reference_cell = _reference_cell_from_npz(data)
        self.crystal_shape = tuple(int(dim) for dim in self.atom_order.shape[:3])
        self.unit_cell_atoms = int(self.atom_order.shape[3])
        self.atom_count = int(np.prod(self.atom_order.shape))
        if self.reference_positions.shape != (self.atom_count, 3):
            raise ValueError("reference_positions shape is inconsistent with atom_order")

        if sequence_length is None:
            sequence_length = int(data["sequence_length"]) if "sequence_length" in data.files else 3
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2 for acceleration models")

        self.centers = build_centers(self.crystal_shape, self.patch_shape, self.periodic)
        self.model = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model_backend = self._detect_model_backend(self.model)
        if int(getattr(self.model, "unit_cell_atoms", self.unit_cell_atoms)) != self.unit_cell_atoms:
            raise ValueError("model unit_cell_atoms does not match dataset atom_order")
        if (
            self.model_backend in {"edge", "pair-force", "pair-energy"}
            and tuple(getattr(self.model, "patch_shape", (3, 3, 3))) != self.patch_shape
        ):
            raise ValueError("Edge/pair ASE inference currently requires patch_shape=(3, 3, 3)")
        if hasattr(self.model, "to"):
            self.model.to(self.device)

        self._history = None
        self._last_flat_displacements = None
        if history_positions is not None:
            self.set_history_positions(history_positions)

        self.low_q_stiffness = None
        self.low_q_mask = None
        self.low_q_mode_count = 0
        self._configure_low_q_correction(data)
        self._force_call_count = 0
        self._curl_cached_antisymmetry = None
        self._history_damping_cached_positive = None
        self._history_damping_power_ema = None
        self._history_damping_diagnostics = []
        self._power_bias_correction_diagnostics = []

    @property
    def history(self):
        """Return a copy of the current crystal-shaped displacement history."""
        if self._history is None:
            return None
        return self._history.copy()

    def set_history_positions(self, positions_history, atoms=None):
        """Initialize the internal history from flat ASE-order positions."""
        positions_history = np.asarray(positions_history, dtype=np.float32)
        expected = (self.sequence_length, self.atom_count, 3)
        if positions_history.shape != expected:
            raise ValueError(f"positions_history must have shape {expected}")
        if atoms is not None:
            self._ensure_reference_cell(atoms)
        displacements = [self._positions_to_crystal_displacements(frame, atoms=atoms) for frame in positions_history]
        self._history = np.asarray(displacements, dtype=np.float32)
        self._last_flat_displacements = self._crystal_to_flat_displacements(self._history[-1])

    def reset_history(self):
        """Clear the internal trajectory history."""
        self._history = None
        self._last_flat_displacements = None

    def replace_current_history_positions(self, positions, atoms=None):
        """Replace the latest history frame with externally corrected positions."""
        current_displacements = self._positions_to_crystal_displacements(positions, atoms=atoms)
        if self._history is None:
            self._history = np.repeat(current_displacements[None, ...], self.sequence_length, axis=0)
        else:
            self._history[-1] = current_displacements
        self._last_flat_displacements = self._crystal_to_flat_displacements(current_displacements)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        """Calculate effective RNN forces for the current ASE atoms."""
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            atoms = self.atoms

        self._validate_atoms(atoms)
        self._ensure_reference_cell(atoms)
        current_displacements = self._positions_to_crystal_displacements(atoms.get_positions(), atoms=atoms)
        self._update_history(current_displacements)

        acceleration = self._predict_acceleration()
        model_energy = None
        if isinstance(acceleration, tuple):
            acceleration, model_energy = acceleration
        acceleration = self._apply_curl_correction(acceleration)
        acceleration = self._apply_history_damping(acceleration)
        acceleration = self._apply_low_q_correction(acceleration)
        acceleration = self._apply_power_bias_correction(acceleration)
        acceleration = self.acceleration_scale * acceleration
        flat_acceleration = self._crystal_to_flat_displacements(acceleration)
        forces = self._acceleration_to_forces(flat_acceleration, atoms)

        self.results["energy"] = self._predict_potential_energy_ev(atoms, model_energy=model_energy)
        self.results["forces"] = forces.astype(np.float64)

    @staticmethod
    def _detect_model_backend(model):
        """Return the calculator backend needed by a saved model object."""
        if getattr(model, "target_mode", None) == "acceleration":
            return "field"
        if getattr(model, "architecture", None) == "pair-energy" and hasattr(model, "predict_full_accelerations"):
            return "pair-energy"
        if getattr(model, "architecture", None) == "pair-force" and hasattr(model, "predict_full_accelerations"):
            return "pair-force"
        if hasattr(model, "edge_features_from_patches") and hasattr(model, "predict_center_accelerations"):
            return "edge"
        raise ValueError(
            "CopperFieldRNNCalculator requires either a target_mode='acceleration' "
            "FieldRNN model, a CrystalEdgeRNNNet model, or a pair-force / pair-energy model"
        )

    def _predict_acceleration(self):
        """Predict one discrete acceleration vector for every crystal atom."""
        if self.model_backend == "field":
            return predict_center_accelerations(
                model=self.model,
                history=self._history,
                centers=self.centers,
                patch_shape=self.patch_shape,
                periodic=self.periodic,
                patch_batch_size=self.patch_batch_size,
                device=self.device,
            )
        if self.model_backend in {"pair-force", "pair-energy"}:
            if self.model_backend == "pair-energy" and hasattr(self.model, "predict_full_accelerations_and_energy"):
                return self.model.predict_full_accelerations_and_energy(
                    self._history,
                    periodic=self.periodic,
                    patch_batch_size=self.patch_batch_size,
                    pair_scatter=True,
                )
            return self.model.predict_full_accelerations(
                self._history,
                periodic=self.periodic,
                patch_batch_size=self.patch_batch_size,
                pair_scatter=True,
            )
        return self._predict_edge_acceleration()

    def _model_energy_to_ev(self, model_energy, atoms):
        """Convert a model-unit pair potential to eV."""
        if self.use_atoms_masses:
            masses = np.asarray(atoms.get_masses(), dtype=np.float64)
            mass_amu = float(np.mean(masses))
        else:
            mass_amu = CU_MASS_AMU
        conversion = mass_amu * AMU_ANGSTROM_PER_PS2_TO_EV_PER_ANGSTROM / (self.dt_ps**2)
        return float(model_energy * conversion)

    def _predict_potential_energy_ev(self, atoms, model_energy=None):
        """Return model potential energy in eV when the backend exposes one."""
        if self.model_backend != "pair-energy" or not hasattr(self.model, "predict_full_potential_energy"):
            return 0.0
        if model_energy is None:
            model_energy = self.model.predict_full_potential_energy(
                self._history,
                periodic=self.periodic,
                patch_batch_size=self.patch_batch_size,
            )
        return self._model_energy_to_ev(model_energy, atoms)

    def _apply_history_damping(self, acceleration):
        """Remove the positive local velocity-response part from acceleration."""
        if self.history_damping_mode == "none":
            return acceleration
        if self.model_backend not in {"field", "edge", "pair-force", "pair-energy"}:
            raise ValueError("history damping currently supports only field, edge, and pair models")
        if not hasattr(self.model, "_center_acceleration_from_patch_tensor"):
            raise ValueError("model does not expose differentiable local patch acceleration")

        corrected = np.asarray(acceleration, dtype=np.float64)
        if self.history_damping_eta > 0:
            if (
                self._history_damping_cached_positive is None
                or (self._force_call_count - 1) % self.history_damping_interval == 0
            ):
                self._history_damping_cached_positive = self._estimate_positive_velocity_response()

            corrected = corrected.copy()
            velocity = np.asarray(self._history[-1] - self._history[-2], dtype=np.float64)
            output_size = self.unit_cell_atoms * 3
            for center_index, center in enumerate(self.centers):
                v_flat = velocity[(*center, slice(None), slice(None))].reshape(output_size)
                correction = -self.history_damping_eta * (
                    self._history_damping_cached_positive[center_index] @ v_flat
                ).reshape(self.unit_cell_atoms, 3)
                corrected[(*center, slice(None), slice(None))] += correction
        corrected = corrected.astype(np.float32)
        self._update_adaptive_history_damping_eta(corrected)
        return corrected

    @property
    def history_damping_diagnostics(self):
        """Return per-force-call adaptive damping diagnostics."""
        return list(self._history_damping_diagnostics)

    @property
    def power_bias_correction_diagnostics(self):
        """Return per-force-call global power-bias correction diagnostics."""
        return list(self._power_bias_correction_diagnostics)

    def _apply_power_bias_correction(self, acceleration):
        """Remove the global acceleration component that injects mean power."""
        if self.power_bias_correction_mode == "none" or self.power_bias_correction_alpha <= 0:
            return acceleration

        velocity = np.asarray(self._history[-1] - self._history[-2], dtype=np.float64)
        acceleration = np.asarray(acceleration, dtype=np.float64)
        power_before = float(np.mean(acceleration * velocity))
        velocity_power = float(np.mean(velocity**2))
        a_rms_before = float(np.sqrt(np.mean(acceleration**2)))
        v_rms = float(np.sqrt(velocity_power))
        normalized_before = power_before / max(
            a_rms_before * v_rms,
            self.power_bias_correction_epsilon,
        )

        if self.power_bias_correction_mode == "global-positive" and power_before <= 0.0:
            lambda_value = 0.0
            corrected = acceleration
        else:
            lambda_value = (
                self.power_bias_correction_alpha
                * power_before
                / max(velocity_power, self.power_bias_correction_epsilon)
            )
            corrected = acceleration - lambda_value * velocity

        power_after = float(np.mean(corrected * velocity))
        a_rms_after = float(np.sqrt(np.mean(corrected**2)))
        normalized_after = power_after / max(
            a_rms_after * v_rms,
            self.power_bias_correction_epsilon,
        )
        self._power_bias_correction_diagnostics.append(
            (
                int(self._force_call_count),
                float(lambda_value),
                power_before,
                power_after,
                normalized_before,
                normalized_after,
                a_rms_before,
                a_rms_after,
                v_rms,
            )
        )
        return corrected.astype(np.float32)

    def _update_adaptive_history_damping_eta(self, acceleration):
        """Slowly adjust history-damping eta from normalized power balance."""
        if self.history_damping_adaptive_gain <= 0:
            return

        velocity = np.asarray(self._history[-1] - self._history[-2], dtype=np.float64)
        acceleration = np.asarray(acceleration, dtype=np.float64)
        power = float(np.mean(acceleration * velocity))
        a_rms = float(np.sqrt(np.mean(acceleration**2)))
        v_rms = float(np.sqrt(np.mean(velocity**2)))
        normalized_power = power / max(a_rms * v_rms, self.history_damping_adaptive_epsilon)

        alpha = self.history_damping_adaptive_ema
        if self._history_damping_power_ema is None:
            self._history_damping_power_ema = normalized_power
        else:
            self._history_damping_power_ema = (
                (1.0 - alpha) * self._history_damping_power_ema + alpha * normalized_power
            )

        old_eta = float(self.history_damping_eta)
        eta = old_eta
        if self._force_call_count % self.history_damping_adaptive_interval == 0:
            error = self._history_damping_power_ema - self.history_damping_adaptive_target_power
            gain = (
                self.history_damping_adaptive_gain
                if error >= 0
                else self.history_damping_adaptive_cooling_gain
            )
            eta = old_eta + gain * error
            eta = max(self.history_damping_adaptive_min_eta, eta)
            if self.history_damping_adaptive_max_eta is not None:
                eta = min(self.history_damping_adaptive_max_eta, eta)
            self.history_damping_eta = float(eta)

        self._history_damping_diagnostics.append(
            (
                int(self._force_call_count),
                old_eta,
                float(self.history_damping_eta),
                power,
                normalized_power,
                float(self._history_damping_power_ema),
                a_rms,
                v_rms,
            )
        )

    def _estimate_positive_velocity_response(self):
        """Estimate the positive symmetric part of ``d a / d(u_t-u_{t-1})``."""
        output_size = self.unit_cell_atoms * 3
        positive_responses = []
        was_training = bool(getattr(self.model.model, "training", False)) if hasattr(self.model, "model") else False
        if hasattr(self.model, "model"):
            self.model.model.eval()
        for start in range(0, len(self.centers), self.history_damping_batch_size):
            batch_centers = self.centers[start : start + self.history_damping_batch_size]
            patch_batch = extract_patch_batch(self._history, batch_centers, self.patch_shape, self.periodic)
            patches = torch.as_tensor(
                patch_batch,
                dtype=torch.float32,
                device=self.device,
            ).detach().clone().requires_grad_(True)
            with torch.backends.cudnn.flags(enabled=False):
                acceleration = self.model._center_acceleration_from_patch_tensor(patches).reshape(
                    patches.shape[0],
                    output_size,
                )
                rows = []
                for output_index in range(output_size):
                    gradient = torch.autograd.grad(
                        acceleration[:, output_index].sum(),
                        patches,
                        create_graph=False,
                        retain_graph=True,
                    )[0]
                    rows.append((-gradient[:, -2, 1, 1, 1]).reshape(patches.shape[0], output_size))
            response = torch.stack(rows, dim=1)
            symmetric = 0.5 * (response + response.transpose(1, 2))
            eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
            positive = torch.clamp(eigenvalues, min=0.0)
            positive_response = (eigenvectors * positive.unsqueeze(1)) @ eigenvectors.transpose(1, 2)
            positive_responses.append(positive_response.detach().cpu().numpy().astype(np.float64))
            del patches, acceleration, rows, response, symmetric, eigenvalues, eigenvectors, positive, positive_response
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        if hasattr(self.model, "model") and was_training:
            self.model.model.train()
        return np.concatenate(positive_responses, axis=0)

    def _apply_curl_correction(self, acceleration):
        """Apply a local linear correction that suppresses Jacobian vorticity."""
        self._force_call_count += 1
        if self.curl_correction_mode == "none" or self.curl_correction_eta <= 0:
            return acceleration
        if self.model_backend not in {"edge", "pair-force", "pair-energy"}:
            raise ValueError("curl correction currently supports only edge and pair models")
        if not hasattr(self.model, "_center_acceleration_from_patch_tensor"):
            raise ValueError("model does not expose differentiable local patch acceleration")

        if (
            self._curl_cached_antisymmetry is None
            or (self._force_call_count - 1) % self.curl_correction_interval == 0
        ):
            self._curl_cached_antisymmetry = self._estimate_local_antisymmetric_jacobians()

        corrected = np.asarray(acceleration, dtype=np.float64).copy()
        displacements = np.asarray(self._history[-1], dtype=np.float64)
        output_size = self.unit_cell_atoms * 3
        for center_index, center in enumerate(self.centers):
            u_flat = displacements[(*center, slice(None), slice(None))].reshape(output_size)
            correction = -self.curl_correction_eta * (
                self._curl_cached_antisymmetry[center_index] @ u_flat
            ).reshape(self.unit_cell_atoms, 3)
            corrected[(*center, slice(None), slice(None))] += correction
        return corrected.astype(np.float32)

    def _estimate_local_antisymmetric_jacobians(self):
        """Estimate local antisymmetric Jacobians d a_center / d u_center."""
        output_size = self.unit_cell_atoms * 3
        antisymmetry = []
        was_training = bool(getattr(self.model.model, "training", False)) if hasattr(self.model, "model") else False
        if hasattr(self.model, "model"):
            self.model.model.eval()
        for start in range(0, len(self.centers), self.curl_correction_batch_size):
            batch_centers = self.centers[start : start + self.curl_correction_batch_size]
            patch_batch = extract_patch_batch(self._history, batch_centers, self.patch_shape, self.periodic)
            patches = torch.as_tensor(
                patch_batch,
                dtype=torch.float32,
                device=self.device,
            ).detach().clone().requires_grad_(True)
            with torch.backends.cudnn.flags(enabled=False):
                acceleration = self.model._center_acceleration_from_patch_tensor(patches).reshape(
                    patches.shape[0],
                    output_size,
                )
                jacobian_rows = []
                for output_index in range(output_size):
                    gradient = torch.autograd.grad(
                        acceleration[:, output_index].sum(),
                        patches,
                        create_graph=False,
                        retain_graph=True,
                    )[0]
                    jacobian_rows.append(gradient[:, -1, 1, 1, 1].reshape(patches.shape[0], output_size))
            jacobian = torch.stack(jacobian_rows, dim=1)
            batch_antisymmetry = 0.5 * (jacobian - jacobian.transpose(1, 2))
            antisymmetry.append(batch_antisymmetry.detach().cpu().numpy().astype(np.float64))
            del patches, acceleration, jacobian_rows, jacobian, batch_antisymmetry
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        if hasattr(self.model, "model") and was_training:
            self.model.model.train()
        return np.concatenate(antisymmetry, axis=0)

    def _predict_edge_acceleration(self):
        """Predict centered accelerations with a CrystalEdgeRNNNet backend."""
        acceleration = np.zeros_like(self._history[-1], dtype=np.float32)
        for start in range(0, len(self.centers), self.patch_batch_size):
            batch_centers = self.centers[start : start + self.patch_batch_size]
            patch_batch = extract_patch_batch(self._history, batch_centers, self.patch_shape, self.periodic)
            center_acceleration = self.model.predict_center_accelerations(patch_batch)
            for local_index, global_center in enumerate(batch_centers):
                acceleration[(*global_center, slice(None), slice(None))] = center_acceleration[local_index]
        return acceleration

    def _configure_low_q_correction(self, data):
        """Prepare optional Fourier-space low-q stiffness correction."""
        if self.low_q_correction_mode == "none" or self.low_q_correction_max_q <= 0:
            return
        if self.reference_cell is None:
            raise ValueError("low-q correction requires a reference cell in data_path")

        q_abs = _q_abs_grid(self.crystal_shape, self.reference_cell)
        mask = q_abs <= self.low_q_correction_max_q
        if self.low_q_correction_exclude_q_zero:
            mask &= q_abs > 0
        if not np.any(mask):
            raise ValueError(
                "low-q correction selected no Fourier modes; increase low_q_correction_max_q "
                "or disable low_q_correction_exclude_q_zero"
            )

        if self.low_q_correction_mode == "reference":
            stiffness = _reference_effective_stiffness(
                data,
                max_frames=self.low_q_reference_max_frames,
                epsilon=self.low_q_stiffness_epsilon,
            )
        else:  # pragma: no cover - guarded by validation above.
            raise ValueError(f"Unsupported low_q_correction_mode={self.low_q_correction_mode!r}")

        if stiffness.shape != tuple(self.crystal_shape):
            raise ValueError("low-q stiffness grid shape does not match crystal_shape")
        self.low_q_stiffness = stiffness
        self.low_q_mask = mask
        self.low_q_mode_count = int(np.count_nonzero(mask))

    def _apply_low_q_correction(self, acceleration):
        """Blend selected low-q acceleration modes toward the reference elastic response."""
        if self.low_q_mask is None or self.low_q_stiffness is None or self.low_q_correction_blend <= 0:
            return acceleration

        cell_count = int(np.prod(self.crystal_shape))
        scale = np.sqrt(cell_count)
        displacement = np.asarray(self._history[-1], dtype=np.float64)
        model_acceleration = np.asarray(acceleration, dtype=np.float64)
        displacement_spectrum = np.fft.fftn(displacement, axes=(0, 1, 2)) / scale
        acceleration_spectrum = np.fft.fftn(model_acceleration, axes=(0, 1, 2)) / scale
        elastic_spectrum = -self.low_q_stiffness[..., None, None] * displacement_spectrum
        mask = self.low_q_mask[..., None, None]
        blend = float(self.low_q_correction_blend)
        corrected_spectrum = np.where(
            mask,
            (1.0 - blend) * acceleration_spectrum + blend * elastic_spectrum,
            acceleration_spectrum,
        )
        corrected = np.fft.ifftn(corrected_spectrum * scale, axes=(0, 1, 2)).real
        return corrected.astype(np.float32)

    def _validate_atoms(self, atoms):
        """Validate that ASE atoms represent the supported Cu crystal."""
        if len(atoms) != self.atom_count:
            raise ValueError(f"Expected {self.atom_count} atoms, got {len(atoms)}")
        atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        if np.any(atomic_numbers != CU_ATOMIC_NUMBER):
            raise ValueError("CopperFieldRNNCalculator only supports Cu atoms")

    def _ensure_reference_cell(self, atoms):
        """Initialize or validate the reference periodic cell."""
        cell = _as_cell_matrix(atoms.get_cell().array)
        if self.reference_cell is None:
            self.reference_cell = cell
            return
        if self.enforce_cell and not np.allclose(cell, self.reference_cell, atol=1e-5, rtol=1e-6):
            raise ValueError("ASE cell differs from the calculator reference cell")

    def _positions_to_crystal_displacements(self, positions, atoms=None):
        """Convert flat ASE-order positions to crystal-shaped displacements."""
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != self.reference_positions.shape:
            raise ValueError("positions shape does not match reference_positions")

        delta = positions - self.reference_positions
        if self.reference_cell is not None:
            pbc = np.ones(3, dtype=bool) if atoms is None else np.asarray(atoms.get_pbc(), dtype=bool)
            fractional = delta @ np.linalg.inv(self.reference_cell)
            fractional[:, pbc] -= np.round(fractional[:, pbc])
            delta = fractional @ self.reference_cell
        return delta.astype(np.float32)[self.atom_order, :]

    def _crystal_to_flat_displacements(self, crystal_values):
        """Convert crystal-shaped atom values back to flat ASE atom order."""
        flat = np.empty((self.atom_count, 3), dtype=np.float32)
        flat[self.atom_order.reshape(-1)] = np.asarray(crystal_values, dtype=np.float32).reshape(self.atom_count, 3)
        return flat

    def _update_history(self, current_displacements):
        """Append the current displacement frame unless this is a repeated force call."""
        current_flat = self._crystal_to_flat_displacements(current_displacements)
        if self._history is None:
            self._history = np.repeat(current_displacements[None, ...], self.sequence_length, axis=0)
            self._last_flat_displacements = current_flat
            return
        if np.allclose(current_flat, self._last_flat_displacements, atol=self.history_tolerance, rtol=0.0):
            self._history[-1] = current_displacements
            return

        self._history[:-1] = self._history[1:]
        self._history[-1] = current_displacements
        self._last_flat_displacements = current_flat

    def _acceleration_to_forces(self, flat_acceleration, atoms):
        """Convert discrete model acceleration to ASE forces in eV/Angstrom."""
        if self.use_atoms_masses:
            masses = np.asarray(atoms.get_masses(), dtype=np.float64)
        else:
            masses = np.full(self.atom_count, CU_MASS_AMU, dtype=np.float64)
        factor = AMU_ANGSTROM_PER_PS2_TO_EV_PER_ANGSTROM / (self.dt_ps**2)
        return flat_acceleration.astype(np.float64) * masses[:, None] * factor
