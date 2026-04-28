# Crystal RNN Repo

This repository contains:

- `base_classes/` with the refactored core classes and physics utilities
- `find_models.py` for training and selecting candidate models

## Dataset

The trajectory dataset `coords_lmp.raw` is distributed as a GitHub Release asset instead of being stored directly in git.

Download it from the `v1.0` release:

- `coords_lmp.raw`: `https://github.com/sinegordon/crystal_rnn/releases/download/v1.0/coords_lmp.raw`

## Crystal Data Preparation

LAMMPS trajectories can be converted to the crystal-shaped format used by `train_crystal_blocks(...)`:

```python
from base_classes import (
    FCC_CONVENTIONAL_BASIS,
    build_crystal_atom_order,
    make_crystal_block_samples,
    positions_to_crystal_displacements,
    read_lammps_dump_positions,
)

positions, box_lengths = read_lammps_dump_positions("Cu.LAMMPSDUMP")
reference_positions = positions[0]
atom_order = build_crystal_atom_order(
    reference_positions=reference_positions,
    crystal_shape=(5, 2, 2),
    box_lengths=box_lengths[0],
    basis_fractional=FCC_CONVENTIONAL_BASIS,
)
displacements = positions_to_crystal_displacements(
    positions=positions,
    reference_positions=reference_positions,
    atom_order=atom_order,
    box_lengths=box_lengths,
)
X_blocks, y_blocks = make_crystal_block_samples(
    displacements=displacements,
    train_supercell_shape=(2, 2, 1),
    sequence_length=3,
)
```

`X_blocks` and `y_blocks` can be passed directly to `model.train_crystal_blocks(...)`.
Displacements are measured from per-atom equilibrium positions estimated as the
mean positions over the loaded trajectory frames.

The same preparation is available as a CLI:

```bash
python prepare_crystal_data.py Cu.LAMMPSDUMP crystal_training_data.npz \
  --input-format dump \
  --crystal-shape 5 2 2 \
  --train-supercell-shape 2 2 1 \
  --sequence-length 3 \
  --start-frame 0 \
  --basis fcc
```

Use `--start-frame N` to skip the first `N` trajectory frames before collecting
data. If `--max-frames M` is also provided, the script reads `M` consecutive
frames after that offset.

## Blockwise Inference

`CrystalRNNNet.run_crystal(...)` runs inference on a larger crystal by sliding the rectangular training supercell over the full displacement field.

Example:

```python
model = CrystalRNNNet(
    hidden_size=100,
    num_layers=3,
    type="GRU",
    train_supercell_shape=(2, 2, 2),
    unit_cell_atoms=4,
    flatten_order=("x", "y", "z", "atom", "coord"),
)

predicted_displacements = model.run_crystal(
    count_steps=100,
    init_displacements=init_displacements,
    stride_shape=(1, 1, 1),
    periodic=False,
)
```

`init_displacements` must have shape:

```python
(sequence_length, nx, ny, nz, unit_cell_atoms, 3)
```

The model input size must match:

```python
prod(train_supercell_shape) * unit_cell_atoms * 3
```

If `stride_shape` is omitted, inference uses `(1, 1, 1)`, so the training supercell moves by one unit cell along each crystal axis. Overlapping predictions for the same `(cell_x, cell_y, cell_z, atom_index)` are averaged before the next rollout step.

`flatten_order` defines how a supercell displacement tensor with shape `(bx, by, bz, unit_cell_atoms, 3)` is packed into the flat model input vector. The default is:

```python
("x", "y", "z", "atom", "coord")
```

The same order is used to unpack model predictions back into the crystal displacement field.

Training can use the same crystal-aware representation:

```python
model.train_crystal_blocks(X_blocks, y_blocks)
```

where:

```python
X_blocks.shape == (n_samples, sequence_length, bx, by, bz, unit_cell_atoms, 3)
y_blocks.shape == (n_samples, bx, by, bz, unit_cell_atoms, 3)
```

`train_crystal_blocks(...)` flattens each training block with the model's `flatten_order`, so training and `run_crystal(...)` share the same packing convention.

`run_crystal(...)` is the physically structured inference interface for rectangular training supercells.
