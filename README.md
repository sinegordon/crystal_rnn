# Crystal RNN Repo

This repository contains:

- `base_classes/` with the refactored core classes and physics utilities
- `find_models.py` for training and selecting candidate models
- `pipelines/` with stable data, search, ASE inference, cluster, and
  postprocessing entrypoints

## Recommended Pipeline

New work should use `pipelines/` rather than ad-hoc root commands.  The current
best-tested workflow is the conservative `pair_energy` path:

1. Prepare force-enabled crystal data from a LAMMPS dump with
   `pipelines/pair_energy/data/prepare_force_data.py`.
2. Search candidate models with
   `pipelines/pair_energy/cluster/submit_search.py`.
3. Check model-search progress with
   `pipelines/pair_energy/cluster/check_search.py`.
4. Run ASE 1055 inference with
   `pipelines/pair_energy/cluster/submit_inference_1055.py`.
5. Fetch lightweight postprocessing outputs with
   `pipelines/pair_energy/cluster/fetch_inference_1055.py`.

The current clean ASE validation mode uses Bussi NVT, `T = 300 K`,
`dt = 0.002 ps`, `taut = 200 fs`, `q_zero_mode = none`, and no internal
history/adaptive/power-bias correctors.  For pair-energy models, direct
`E_tot = E_pot_model + E_kin` is the primary energy diagnostic; path-energy is
kept as a force/work consistency diagnostic.

See [pipelines/README.md](pipelines/README.md) and
[pipelines/pair_energy/README.md](pipelines/pair_energy/README.md) for full
commands.

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

## ASE Copper Calculator

The acceleration FieldRNN model can also be used as a stateful ASE force
calculator for Cu crystals.  It keeps the recent displacement history internally
and converts model-predicted discrete accelerations to ASE forces:

```python
from ase.io import read

from ase_copper_calculator import CopperFieldRNNCalculator

atoms = read("cu_start.xyz")
history_positions = ...  # shape: (sequence_length, len(atoms), 3), Angstrom

atoms.calc = CopperFieldRNNCalculator(
    model_path="models333_field_rnn_accnorm_cl2_d30k_ep100/mean_norm_0.9217232867679755_rnn_field_rnn_acceleration_ec32_rh64_rl1_bidir_accnormchannel_cl2_k3.pth",
    data_path="data1055.npz",
    history_positions=history_positions,
    dt_ps=0.02,
    patch_shape=(3, 3, 3),
    periodic=True,
    device="cuda",
)

forces = atoms.get_forces()
```

This wrapper reports a dummy potential energy of `0.0`, because the current
network is not an energy model.  For real MD runs, initialize
`history_positions` from consecutive frames or from positions consistent with
the intended initial velocity; otherwise the calculator starts from repeated
current positions.

Run an ASE NVT trajectory with the same calculator:

```bash
python run_ase_copper_nvt.py \
  --model-path models333_field_rnn_accnorm_cl2_d30k_ep100/mean_norm_0.9217232867679755_rnn_field_rnn_acceleration_ec32_rh64_rl1_bidir_accnormchannel_cl2_k3.pth \
  --data-path data1055.npz \
  --output-npz inference_outputs/ase_nvt/cu_nvt.npz \
  --trajectory-path inference_outputs/ase_nvt/cu_nvt.traj \
  --steps 2000 \
  --initial-frames 0 1 2 \
  --temperature-k 300 \
  --taut-fs 200 \
  --dt-ps 0.02 \
  --patch-shape 3 3 3 \
  --device cuda
```

`--initial-frames` selects the three starting frames from the prepared `.npz`.
The script derives initial velocities from the last two frames and then runs
ASE Bussi NVT.  Add `--rescale-initial-temperature` if you want to rescale
those frame-derived velocities to `--temperature-k` before starting.

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
    target_mode="absolute",
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

`target_mode` defines what the RNN learns:

```python
target_mode="absolute"  # learn the next displacement field directly
target_mode="delta"     # learn next_displacement - last_input_displacement
target_mode="absolute_delta"  # learn absolute output plus an auxiliary normalized delta loss
target_mode="acceleration"  # learn next - 2 * last_input + previous_input
target_mode="verlet"  # output acceleration, train reconstructed next displacement
```

For `target_mode="delta"`, `run_crystal(...)` automatically adds the predicted
delta back to the last input frame before the next autoregressive step. This is
useful for blockwise inference because overlapping blocks stitch predicted
changes instead of independently stitched absolute coordinates.

For `target_mode="absolute_delta"`, inference is the same as `absolute`: the
model output is the next displacement field. Training adds an extra normalized
delta penalty:

```python
true_delta = true_next - last_input
pred_delta = pred_next - last_input
delta_scale = rms(true_delta).clip(min=delta_loss_epsilon)
loss = MSE(pred_next, true_next) + delta_loss_weight * MSE(pred_delta / delta_scale, true_delta / delta_scale)
```

This keeps the absolute target scale while giving the one-step change its own
relative weight. A non-normalized `MSE(pred_delta, true_delta)` would be
mathematically identical to `MSE(pred_next, true_next)`, because the same
`last_input` is subtracted from both sides.

Training can also include an optional normalized discrete-acceleration penalty:

```python
pred_acceleration = pred_next - 2 * last_input + previous_input
true_acceleration = true_next - 2 * last_input + previous_input
acceleration_scale = rms(true_acceleration).clip(min=acceleration_loss_epsilon)
loss += acceleration_loss_weight * MSE(
    pred_acceleration / acceleration_scale,
    true_acceleration / acceleration_scale,
)
```

This is disabled by default. A conservative first experiment is
`--acceleration-loss-weight 1e-4` or `1e-3`.

For `target_mode="acceleration"`, the RNN output is the discrete acceleration
itself. Inference reconstructs the next displacement with the Verlet-like step:

```python
next_displacement = 2 * last_input - previous_input + predicted_acceleration
```

For `target_mode="verlet"`, the RNN output is also a discrete acceleration, but
the primary training loss is applied to the reconstructed next displacement.
The auxiliary normalized delta and acceleration losses can still be enabled.

`find_models.py` can train on short local rollouts instead of only one-step
samples:

```bash
python find_models.py 10 RNN \
  --data-path data333.npz \
  --models-dir models333_rollout3 \
  --target-mode absolute_delta \
  --delta-loss-weight 3 \
  --rollout-steps 3
```

For a stronger but more expensive test, use `--rollout-steps 5`.

`run_crystal(...)` is the physically structured inference interface for rectangular training supercells.

## Saved Model Inference

Use `infer_model.py` to run a model selected by `find_models.py`:

```bash
python infer_model.py \
  --model-path models/mean_norm_1.8874629875305242_rrn_crystal_400.pth \
  --data-path data.npz \
  --output-path prediction.npz \
  --count-steps 2000 \
  --start-frame 5000 \
  --merge-mode mean \
  --reference-output \
  --save-positions
```

`--start-frame` selects the first frame of the initial history. If the training
sequence length is `3`, `--start-frame 5000` uses frames `5000, 5001, 5002` as
input and starts prediction at frame `5003`.

The output always contains `predicted_displacements`, `init_displacements`,
`reference_positions`, `atom_order`, and basic inference metadata. With
`--reference-output`, it also saves the real continuation from the dataset. With
`--save-positions`, it saves absolute flat-position arrays in the original atom
order for predicted, initial, and reference frames when available.

`--merge-mode` controls how overlapping block predictions are stitched:
`mean` averages overlaps, `weighted` gives central block cells larger weights,
`center` assigns each crystal cell to the block where it is closest to the
block center, and `owner` assigns each crystal cell to the first block that
covers it. `robust_center` and `robust_mean` compare all block predictions for
the same physical cell against their median and penalize candidates farther
from the block center:

```text
score = distance_to_median + merge_alpha * distance_from_block_center
```

`robust_center` chooses the best candidate. `robust_mean` averages the best
`--merge-top-k` candidates, defaulting to three candidates when `--merge-top-k`
is omitted.

Plot predicted and reference S(q,w) maps from the inference output:

```bash
python plot_sqw_comparison.py \
  --input-path prediction.npz \
  --output-path sqw_comparison.png
```

The figure contains predicted and reference maps side by side with a shared
color scale. The script also prints the Pearson correlation between the two
S(q,w) intensity maps.

## Blockwise Fine-Tuning

Use `finetune_blockwise.py` to continue training a saved small-block model
through the same stitched blockwise rollout used for large-crystal inference:

```bash
python finetune_blockwise.py \
  --model-path models/model.pth \
  --data-path data1055.npz \
  --output-model-path models/model_blockwise_ft.pth \
  --merge-mode owner \
  --rollout-steps 3 \
  --epochs 2 \
  --steps-per-epoch 20 \
  --learning-rate 1e-5
```

`--rollout-steps` controls how many autoregressive steps remain in the
differentiable training graph. Values above one make training closer to the
long-run inference task, but also make each optimization step more expensive.

## ConvRNN Models

`CrystalConvRNNNet` is an alternative architecture that keeps the crystal as a
3D grid instead of flattening each supercell.  The channel dimension is
`unit_cell_atoms * 3`, so an FCC crystal has 12 input/output channels.  The same
model can be trained on a small block and run directly on a larger crystal,
because the recurrent operations are 3D convolutions.

Search ConvGRU models with the S(q,w)-based selector:

```bash
python find_conv_models.py \
  10 ConvGRU \
  --data-path data333.npz \
  --models-dir models_conv333 \
  --target-mode absolute_delta \
  --delta-loss-weight 3 \
  --hidden-channels 32 \
  --num-layers 2 \
  --kernel-size 3 \
  --periodic-padding \
  --residual-output
```

Supported recurrent cells are `ConvRNN`, `ConvGRU`, and `ConvLSTM`.  Use
`--no-periodic-padding` to replace circular convolution padding with zero
padding.  `--residual-output` makes the convolutional network predict a
correction to the last input frame; the training target can still stay
`absolute` or `absolute_delta`.

`find_field_rnn_models.py` supports auxiliary losses for acceleration models.
The low-q stiffness loss compares the Fourier-space acceleration response per
displacement amplitude on the lowest crystal modes:

```bash
python find_field_rnn_models.py 10 RNN \
  --data-path data333.npz \
  --target-mode acceleration \
  --acceleration-normalization channel \
  --low-q-stiffness-loss-weight 0.1 \
  --low-q-stiffness-max-shell 1
```

The default `--low-q-stiffness-max-shell 1` uses the six axis modes
`n_x^2+n_y^2+n_z^2 = 1`, excluding the zero mode.

## Hybrid Flat + Conv Models

`CrystalHybridRNNNet` mixes the legacy flat block RNN with a ConvRNN branch:

```text
prediction = alpha * flat_prediction + (1 - alpha) * conv_prediction
```

`alpha` is trainable.  During large-crystal inference the ConvRNN branch predicts
the full crystal, while the flat RNN branch predicts training-size blocks; the
two predictions are mixed inside each block and then stitched with the selected
merge mode.

Search hybrid models:

```bash
python find_hybrid_models.py \
  10 \
  --data-path data333.npz \
  --models-dir models_hybrid333 \
  --flat-type RNN \
  --flat-hidden-size 100 \
  --flat-num-layers 3 \
  --conv-type ConvGRU \
  --conv-hidden-channels 32 \
  --conv-num-layers 1 \
  --target-mode absolute_delta \
  --delta-loss-weight 3 \
  --merge-mode owner
```

To use a previously selected flat RNN as the stable branch and train only the
ConvRNN correction plus mixture weight:

```bash
python find_hybrid_models.py \
  10 \
  --data-path data333.npz \
  --models-dir models_hybrid333 \
  --pretrained-flat-model models333_delta/model.pth \
  --freeze-flat \
  --conv-type ConvGRU \
  --conv-hidden-channels 32 \
  --conv-num-layers 1 \
  --target-mode absolute_delta \
  --delta-loss-weight 3 \
  --residual-output \
  --initial-alpha 0.9 \
  --merge-mode owner
```

Compare atom ordering between two prepared datasets:

```bash
python check_atom_order.py data333.npz data1055.npz
```

The script reports per-slot local fractional basis positions, periodic spread,
and a suggested `atom_in_cell` reorder index if the two datasets use different
basis-slot ordering.
