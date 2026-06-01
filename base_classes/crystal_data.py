import numpy as np


AMU_ANGSTROM_PER_PS2_TO_EV_PER_ANGSTROM = 1.0364269656262175e-4
"""Conversion factor from ``amu * Angstrom / ps^2`` to ``eV / Angstrom``."""

CU_MASS_AMU = 63.546
"""Default copper atomic mass used for force-to-acceleration conversion."""


FCC_CONVENTIONAL_BASIS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ],
    dtype=np.float32,
)
"""Fractional basis positions for a conventional FCC unit cell."""


def read_raw_positions(path, max_frames=None, start_frame=0):
    """Read whitespace-separated flat coordinate frames.

    Args:
        path: Path to a `.raw` file where each row contains `x y z` triples.
        max_frames: Optional maximum number of frames to read.
        start_frame: Number of initial frames to skip before reading.

    Returns:
        Array with shape `(frames, atoms, 3)`.
    """
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")

    frames = []
    with open(path) as file:
        for line_index, line in enumerate(file):
            if line_index < start_frame:
                continue
            if max_frames is not None and len(frames) >= max_frames:
                break
            values = np.fromstring(line, sep=" ", dtype=np.float32)
            if values.size % 3 != 0:
                raise ValueError("Each raw coordinate row must contain complete x y z triples")
            frames.append(values.reshape(-1, 3))

    if not frames:
        raise ValueError("No coordinate frames were read")
    return np.stack(frames, axis=0)


def read_lammps_dump_positions(path, max_frames=None, sort_by_id=True, start_frame=0):
    """Read atom positions from a LAMMPS dump trajectory.

    The parser expects `ITEM: ATOMS` records with `id`, `x`, `y`, and `z`
    columns. Orthorhombic box lengths are returned from the dump bounds; tilt
    factors in triclinic headers are ignored for now.

    Args:
        path: Path to a LAMMPS dump file.
        max_frames: Optional maximum number of frames to read.
        sort_by_id: Whether to sort atoms by atom id inside each frame.
        start_frame: Number of initial frames to skip before collecting frames.

    Returns:
        A tuple `(positions, box_lengths)` where positions has shape
        `(frames, atoms, 3)` and box_lengths has shape `(frames, 3)`.
    """
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")

    arrays = read_lammps_dump_arrays(
        path,
        max_frames=max_frames,
        sort_by_id=sort_by_id,
        start_frame=start_frame,
        read_forces=False,
    )
    return arrays["positions"], arrays["box_lengths"]


def read_lammps_dump_arrays(path, max_frames=None, sort_by_id=True, start_frame=0, read_forces=True, require_forces=False):
    """Read positions and optional forces from a LAMMPS dump trajectory.

    Args:
        path: Path to a LAMMPS dump file.
        max_frames: Optional maximum number of frames to read.
        sort_by_id: Whether to sort atoms by atom id inside each frame.
        start_frame: Number of initial frames to skip before collecting frames.
        read_forces: If true, read ``fx fy fz`` when those columns are present.
        require_forces: If true, raise an error when force columns are absent.

    Returns:
        Dictionary with ``positions`` and ``box_lengths`` arrays.  When force
        columns are available and requested, ``forces_ev_per_ang`` is also
        present with shape ``(frames, atoms, 3)``.
    """
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")

    frames = []
    boxes = []
    forces_frames = []
    saw_force_columns = False
    seen_frames = 0

    with open(path) as file:
        while max_frames is None or len(frames) < max_frames:
            line = file.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            file.readline()
            _expect_header(file.readline(), "ITEM: NUMBER OF ATOMS")
            atom_count = int(file.readline())
            box_header = file.readline()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Expected ITEM: BOX BOUNDS section")

            bounds = []
            for _ in range(3):
                parts = [float(value) for value in file.readline().split()]
                bounds.append(parts[:2])
            box_lengths = np.array([high - low for low, high in bounds], dtype=np.float32)

            atom_header = file.readline().split()
            if atom_header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("Expected ITEM: ATOMS section")
            columns = atom_header[2:]
            id_column = columns.index("id") if "id" in columns else None
            x_column = columns.index("x")
            y_column = columns.index("y")
            z_column = columns.index("z")
            force_columns = None
            if read_forces and all(name in columns for name in ("fx", "fy", "fz")):
                force_columns = (columns.index("fx"), columns.index("fy"), columns.index("fz"))
                saw_force_columns = True
            elif require_forces:
                raise ValueError("LAMMPS dump does not contain fx fy fz columns")

            ids = []
            positions = []
            forces = []
            for _ in range(atom_count):
                values = file.readline().split()
                if id_column is not None:
                    ids.append(int(values[id_column]))
                positions.append(
                    [
                        float(values[x_column]),
                        float(values[y_column]),
                        float(values[z_column]),
                    ]
                )
                if force_columns is not None:
                    forces.append(
                        [
                            float(values[force_columns[0]]),
                            float(values[force_columns[1]]),
                            float(values[force_columns[2]]),
                        ]
                    )

            positions = np.asarray(positions, dtype=np.float32)
            if force_columns is not None:
                forces = np.asarray(forces, dtype=np.float32)
            if sort_by_id and id_column is not None:
                order = np.argsort(np.asarray(ids))
                positions = positions[order]
                if force_columns is not None:
                    forces = forces[order]

            if seen_frames >= start_frame:
                frames.append(positions)
                boxes.append(box_lengths)
                if force_columns is not None:
                    forces_frames.append(forces)
            seen_frames += 1

    if not frames:
        raise ValueError("No LAMMPS frames were read")
    result = {
        "positions": np.stack(frames, axis=0),
        "box_lengths": np.stack(boxes, axis=0),
    }
    if forces_frames:
        if len(forces_frames) != len(frames):
            raise ValueError("Force columns are present only for part of the selected dump frames")
        result["forces_ev_per_ang"] = np.stack(forces_frames, axis=0)
    elif read_forces and require_forces and not saw_force_columns:
        raise ValueError("No force columns were read from the LAMMPS dump")
    return result


def build_crystal_atom_order(reference_positions, crystal_shape, box_lengths, unit_cell_atoms=None, basis_fractional=None):
    """Map atom indices to `(cell_x, cell_y, cell_z, atom_in_cell)` slots.

    If `basis_fractional` is provided, each atom is assigned to the nearest
    `(unit cell, basis atom)` slot with periodic wrapping. This is the preferred
    mode for thermal LAMMPS frames where atoms near a boundary may be wrapped
    into the neighboring periodic image. Without `basis_fractional`, atoms are
    assigned by flooring their fractional unit-cell coordinates.

    Args:
        reference_positions: Reference positions with shape `(atoms, 3)`.
        crystal_shape: Number of unit cells along `(x, y, z)`.
        box_lengths: Orthorhombic simulation box lengths.
        unit_cell_atoms: Number of atoms in one unit cell. Inferred from
            `basis_fractional` when omitted.
        basis_fractional: Optional basis coordinates inside one unit cell.

    Returns:
        Integer array with shape `(nx, ny, nz, unit_cell_atoms)` containing
        atom indices from `reference_positions`.
    """
    crystal_shape = _shape3(crystal_shape, "crystal_shape")
    box_lengths = np.asarray(box_lengths, dtype=np.float32)
    reference_positions = np.asarray(reference_positions, dtype=np.float32)
    if basis_fractional is not None:
        basis_fractional = np.asarray(basis_fractional, dtype=np.float32)
        if basis_fractional.ndim != 2 or basis_fractional.shape[1] != 3:
            raise ValueError("basis_fractional must have shape (unit_cell_atoms, 3)")
        if unit_cell_atoms is None:
            unit_cell_atoms = basis_fractional.shape[0]
        if basis_fractional.shape[0] != unit_cell_atoms:
            raise ValueError("basis_fractional length does not match unit_cell_atoms")

    if unit_cell_atoms is None:
        raise ValueError("unit_cell_atoms is required when basis_fractional is omitted")

    expected_atoms = int(np.prod(crystal_shape) * unit_cell_atoms)
    if reference_positions.shape != (expected_atoms, 3):
        raise ValueError("reference_positions shape does not match crystal_shape and unit_cell_atoms")

    scaled = (reference_positions % box_lengths) / box_lengths * np.asarray(crystal_shape)
    if basis_fractional is not None:
        return _build_basis_atom_order(scaled, crystal_shape, basis_fractional)

    cell_indices = np.floor(scaled).astype(int) % np.asarray(crystal_shape)
    local_fractional = scaled - np.floor(scaled)

    buckets = {}
    for atom_index, cell_index in enumerate(cell_indices):
        buckets.setdefault(tuple(cell_index), []).append(atom_index)

    atom_order = np.empty((*crystal_shape, unit_cell_atoms), dtype=np.int64)
    for cell_index, atom_indices in buckets.items():
        if len(atom_indices) != unit_cell_atoms:
            raise ValueError(f"Cell {cell_index} contains {len(atom_indices)} atoms, expected {unit_cell_atoms}")
        atom_indices = np.asarray(atom_indices, dtype=np.int64)
        local = local_fractional[atom_indices]
        local_order = np.lexsort((local[:, 2], local[:, 1], local[:, 0]))
        atom_order[cell_index] = atom_indices[local_order]

    if len(buckets) != int(np.prod(crystal_shape)):
        raise ValueError("Not all crystal cells were populated")
    return atom_order


def _build_basis_atom_order(scaled_positions, crystal_shape, basis_fractional):
    unit_cell_atoms = basis_fractional.shape[0]
    atom_order = np.full((*crystal_shape, unit_cell_atoms), -1, dtype=np.int64)
    crystal_shape_array = np.asarray(crystal_shape, dtype=np.float32)

    for atom_index, scaled_position in enumerate(scaled_positions):
        best_cell = None
        best_basis_index = None
        best_distance = None

        for basis_index, basis_position in enumerate(basis_fractional):
            cell_float = scaled_position - basis_position
            cell_index = np.rint(cell_float).astype(int)
            ideal = cell_index + basis_position
            delta = scaled_position - ideal
            delta -= crystal_shape_array * np.round(delta / crystal_shape_array)
            distance = float(np.dot(delta, delta))

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_cell = tuple((cell_index % np.asarray(crystal_shape)).astype(int))
                best_basis_index = basis_index

        if atom_order[best_cell][best_basis_index] != -1:
            raise ValueError(f"Duplicate atom assignment for cell {best_cell}, basis {best_basis_index}")
        atom_order[best_cell][best_basis_index] = atom_index

    if np.any(atom_order < 0):
        raise ValueError("Not all crystal cell/basis slots were populated")
    return atom_order


def positions_to_crystal_displacements(positions, reference_positions, atom_order, box_lengths):
    """Convert positions to crystal-shaped displacements from equilibrium.

    Args:
        positions: Position trajectory with shape `(frames, atoms, 3)`.
        reference_positions: Equilibrium positions with shape `(atoms, 3)`.
        atom_order: Mapping from `build_crystal_atom_order`.
        box_lengths: Orthorhombic simulation box lengths.

    Returns:
        Displacements with shape `(frames, nx, ny, nz, unit_cell_atoms, 3)`.
    """
    positions = np.asarray(positions, dtype=np.float32)
    reference_positions = np.asarray(reference_positions, dtype=np.float32)
    box_lengths = np.asarray(box_lengths, dtype=np.float32)
    if box_lengths.ndim == 2:
        box_lengths = box_lengths[0]

    displacements = positions - reference_positions[None, :, :]
    displacements -= box_lengths * np.round(displacements / box_lengths)
    return displacements[:, atom_order, :]


def flat_vectors_to_crystal_values(values, atom_order):
    """Convert flat atom-vector frames to crystal-shaped vectors.

    This is used for true vector quantities such as forces.  Unlike
    displacements, vectors are not minimum-image wrapped and do not use a
    reference frame.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("values must have shape (frames, atoms, 3)")
    atom_order = np.asarray(atom_order, dtype=np.int64)
    if values.shape[1] != atom_order.size:
        raise ValueError("values atom count does not match atom_order")
    return values[:, atom_order, :]


def forces_to_discrete_accelerations(forces_ev_per_ang, dt_ps, atom_mass_amu=CU_MASS_AMU):
    """Convert forces in ``eV/Angstrom`` to model discrete accelerations.

    The RNN/ASE integration uses a Verlet-like update,

        ``u[n + 1] = 2*u[n] - u[n - 1] + a_discrete[n]``.

    For forces from LAMMPS ``units metal`` the corresponding discrete
    acceleration is ``F * dt_ps**2 / (mass * conversion_factor)``.
    """
    if dt_ps <= 0:
        raise ValueError("dt_ps must be positive")
    if atom_mass_amu <= 0:
        raise ValueError("atom_mass_amu must be positive")
    forces_ev_per_ang = np.asarray(forces_ev_per_ang, dtype=np.float32)
    scale = float(dt_ps) ** 2 / (float(atom_mass_amu) * AMU_ANGSTROM_PER_PS2_TO_EV_PER_ANGSTROM)
    return (forces_ev_per_ang * scale).astype(np.float32)


def make_crystal_block_samples(
    displacements,
    train_supercell_shape,
    sequence_length,
    stride_shape=None,
    periodic=False,
    target_fields=None,
    target_frame_offset=None,
):
    """Create crystal block samples for `CrystalRNNNet.train_crystal_blocks`.

    Args:
        displacements: Full crystal displacements with shape
            `(frames, nx, ny, nz, unit_cell_atoms, 3)`.
        train_supercell_shape: Training supercell shape `(bx, by, bz)`.
        sequence_length: Number of history frames per input sample.
        stride_shape: Origin stride in unit-cell coordinates. Defaults to
            `(1, 1, 1)`.
        periodic: Whether supercell blocks may wrap around crystal boundaries.
        target_fields: Optional frame-aligned crystal fields used for output
            blocks instead of next-frame displacements.
        target_frame_offset: Frame offset from each input-window start used
            for ``target_fields``.  For force-derived Verlet acceleration
            targets this should be ``sequence_length - 1``.

    Returns:
        A tuple `(X_blocks, y_blocks)` ready for `train_crystal_blocks`.
    """
    displacements = np.asarray(displacements, dtype=np.float32)
    train_supercell_shape = _shape3(train_supercell_shape, "train_supercell_shape")
    stride_shape = (1, 1, 1) if stride_shape is None else _shape3(stride_shape, "stride_shape")
    crystal_shape = tuple(displacements.shape[1:4])
    origins = _build_origins(crystal_shape, train_supercell_shape, stride_shape, periodic)
    target_fields = displacements if target_fields is None else np.asarray(target_fields, dtype=np.float32)
    if target_frame_offset is None:
        target_frame_offset = sequence_length
    target_frame_offset = int(target_frame_offset)
    if target_frame_offset < 0:
        raise ValueError("target_frame_offset must be non-negative")
    if target_fields.shape[1:] != displacements.shape[1:]:
        raise ValueError("target_fields crystal shape must match displacements")
    sample_windows = min(displacements.shape[0] - sequence_length, target_fields.shape[0] - target_frame_offset)
    if sample_windows <= 0:
        raise ValueError("Need more frames than sequence_length")

    X_blocks = []
    y_blocks = []
    for time_index in range(sample_windows):
        history = displacements[time_index : time_index + sequence_length]
        target = target_fields[time_index + target_frame_offset]
        for origin in origins:
            index = _supercell_index(origin, train_supercell_shape, crystal_shape, periodic)
            X_blocks.append(history[(slice(None), *index, slice(None), slice(None))])
            y_blocks.append(target[(*index, slice(None), slice(None))])

    return np.asarray(X_blocks, dtype=np.float32), np.asarray(y_blocks, dtype=np.float32)


def _expect_header(line, expected):
    if not line.startswith(expected):
        raise ValueError(f"Expected {expected}")


def _shape3(value, name):
    if len(value) != 3:
        raise ValueError(f"{name} must contain three dimensions")
    value = tuple(int(dim) for dim in value)
    if any(dim <= 0 for dim in value):
        raise ValueError(f"{name} dimensions must be positive")
    return value


def _build_origins(crystal_shape, block_shape, stride_shape, periodic):
    if periodic:
        return [
            (ix, iy, iz)
            for ix in range(0, crystal_shape[0], stride_shape[0])
            for iy in range(0, crystal_shape[1], stride_shape[1])
            for iz in range(0, crystal_shape[2], stride_shape[2])
        ]
    return [
        (ix, iy, iz)
        for ix in range(0, crystal_shape[0] - block_shape[0] + 1, stride_shape[0])
        for iy in range(0, crystal_shape[1] - block_shape[1] + 1, stride_shape[1])
        for iz in range(0, crystal_shape[2] - block_shape[2] + 1, stride_shape[2])
    ]


def _supercell_index(origin, block_shape, crystal_shape, periodic):
    axes = []
    for start, block_dim, crystal_dim in zip(origin, block_shape, crystal_shape):
        values = np.arange(start, start + block_dim)
        if periodic:
            values = values % crystal_dim
        axes.append(values)
    return np.ix_(axes[0], axes[1], axes[2])
