"""Train several crystal-aware RNN models and keep the best candidates.

The script expects training data prepared by ``prepare_crystal_data.py``.  The
data contains crystal-shaped displacement tensors for training and the original
flat atom order required by the legacy S(q,w) evaluation utility.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from base_classes import CrystalRNNNet, DEFAULT_FLATTEN_ORDER, get_sqw
from base_classes.crystal_predictor import (
    ORDER_AXIS_TO_DIM,
    _build_supercell_origins,
    _normalize_center_loss_alpha,
    _normalize_center_loss_weight,
    _normalize_delta_loss_epsilon,
    _normalize_delta_loss_weight,
    _normalize_loss_weight_mode,
    _normalize_target_mode,
    _weighted_mse_loss,
)


COUNT_STEPS = 2000
DELTA = 10000
STEP = 10
DT = 0.02
HIDDEN_SIZE = 100
DATA_LEN = 0.2
BATCH_SIZE = 200
NUM_LAYERS = 3
COUNT_RUN = 3
SAVE_THRESHOLD = 4.5
LATTICE_PARAMETER = 3.615
NCELLS = 3
KCOUNT = 3


def parse_args():
    """Parse command-line options for model search and evaluation."""
    parser = argparse.ArgumentParser(description="Train and select crystal-aware RNN predictors.")
    parser.add_argument("count_models", type=int, help="Number of models to train.")
    parser.add_argument("rnn_type", help="Recurrent block type: RNN, GRU, or LSTM.")
    parser.add_argument(
        "--data-path",
        default="crystal_training_data.npz",
        help="Path to an .npz file from prepare_crystal_data.py.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory where selected models will be saved.",
    )
    parser.add_argument("--count-steps", type=int, default=COUNT_STEPS)
    parser.add_argument("--count-run", type=int, default=COUNT_RUN)
    parser.add_argument("--save-threshold", type=float, default=SAVE_THRESHOLD)
    parser.add_argument("--delta-frames", type=int, default=DELTA)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument(
        "--temporal-architecture",
        choices=["stacked", "frame-layered"],
        default="stacked",
        help=(
            "Flat temporal network. 'stacked' uses PyTorch nn.RNN/GRU/LSTM num_layers; "
            "'frame-layered' uses one separate recurrent cell per history frame."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data-len", type=float, default=DATA_LEN)
    parser.add_argument(
        "--periodic",
        action="store_true",
        help="Use periodic wrapping during crystal inference.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["absolute", "delta", "absolute_delta", "acceleration", "verlet"],
        default="absolute",
        help=(
            "Training target convention. 'absolute' learns the next displacement; "
            "'delta' learns next displacement minus the last input frame; "
            "'absolute_delta' learns absolute displacement with an auxiliary normalized delta loss; "
            "'acceleration' learns next minus 2*last plus previous; "
            "'verlet' outputs acceleration but trains the reconstructed next displacement."
        ),
    )
    parser.add_argument(
        "--delta-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the auxiliary delta loss when --target-mode absolute_delta.",
    )
    parser.add_argument(
        "--delta-loss-epsilon",
        type=float,
        default=1e-6,
        help="Small stabilizer for normalized delta loss when --target-mode absolute_delta.",
    )
    parser.add_argument(
        "--acceleration-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the auxiliary normalized acceleration loss.",
    )
    parser.add_argument(
        "--acceleration-loss-epsilon",
        type=float,
        default=1e-8,
        help="Small stabilizer for normalized acceleration loss.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1,
        help="Number of differentiable local autoregressive training steps. 1 keeps the original one-step training.",
    )
    parser.add_argument(
        "--rollout-loss-decay",
        type=float,
        default=1.0,
        help="Per-step multiplier for multi-step rollout losses.",
    )
    parser.add_argument(
        "--loss-weight-mode",
        choices=["uniform", "center", "soft_center"],
        default="uniform",
        help=(
            "Output weighting for training loss. 'center' upweights the geometrical center cell; "
            "'soft_center' decays smoothly away from the center."
        ),
    )
    parser.add_argument(
        "--center-loss-weight",
        type=float,
        default=1.0,
        help="Multiplier for the central cell when --loss-weight-mode is not uniform.",
    )
    parser.add_argument(
        "--center-loss-alpha",
        type=float,
        default=1.0,
        help="Distance decay for --loss-weight-mode soft_center.",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Save an S(q,w) comparison image for every evaluated model.",
    )
    parser.add_argument(
        "--plot-output-dir",
        default=None,
        help="Directory for per-iteration S(q,w) comparison images.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show each S(q,w) image interactively after saving. This blocks until the plot window is closed.",
    )
    return parser.parse_args()


def load_training_data(path):
    """Load and validate crystal training arrays from an ``.npz`` dataset."""
    print("IMPORT DATA")
    data = np.load(path)
    required = [
        "X_blocks",
        "y_blocks",
        "displacements",
        "atom_order",
        "reference_positions",
        "crystal_shape",
        "train_supercell_shape",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {missing}")

    loaded = {key: data[key] for key in data.files}
    print("X_blocks", loaded["X_blocks"].shape)
    print("y_blocks", loaded["y_blocks"].shape)
    print("displacements", loaded["displacements"].shape)
    return loaded


def sample_train_data(data, delta, sequence_length):
    """Select a random consecutive frame window and its training blocks."""
    displacements = data["displacements"]
    X_blocks = data["X_blocks"]
    y_blocks = data["y_blocks"]
    frame_count = displacements.shape[0]
    total_time_windows = frame_count - sequence_length
    if total_time_windows <= 0:
        raise ValueError("Need more displacement frames than sequence_length")
    if X_blocks.shape[0] != y_blocks.shape[0]:
        raise ValueError("X_blocks and y_blocks must contain the same number of samples")
    if X_blocks.shape[0] % total_time_windows != 0:
        raise ValueError("X_blocks count is inconsistent with displacements and sequence_length")

    blocks_per_time_window = X_blocks.shape[0] // total_time_windows
    if frame_count <= delta:
        start_frame = 0
        window_frame_count = frame_count
    else:
        start_frame = np.random.randint(low=0, high=frame_count - delta + 1)
        window_frame_count = delta

    start_sample = start_frame * blocks_per_time_window
    sample_count = (window_frame_count - sequence_length) * blocks_per_time_window
    if sample_count <= 0:
        raise ValueError("Selected training window is too short for sequence_length")

    stop_sample = start_sample + sample_count
    print(f"TRAIN FRAMES {start_frame}:{start_frame + window_frame_count}")
    print(f"TRAIN BLOCKS {start_sample}:{stop_sample}")
    train_displacements = displacements[start_frame : start_frame + window_frame_count]
    return train_displacements, X_blocks[start_sample:stop_sample], y_blocks[start_sample:stop_sample]


def torch_flatten_supercell(block, flatten_order):
    """Pack a torch supercell tensor into the model's flat feature order."""
    axes = [ORDER_AXIS_TO_DIM[name] for name in flatten_order]
    return block.permute(*axes).reshape(-1)


def torch_unflatten_supercell(flat_features, train_supercell_shape, unit_cell_atoms, flatten_order):
    """Restore one flat torch vector to canonical crystal block shape."""
    ordered_shape = []
    for name in flatten_order:
        if name == "x":
            ordered_shape.append(train_supercell_shape[0])
        elif name == "y":
            ordered_shape.append(train_supercell_shape[1])
        elif name == "z":
            ordered_shape.append(train_supercell_shape[2])
        elif name == "atom":
            ordered_shape.append(unit_cell_atoms)
        elif name == "coord":
            ordered_shape.append(3)

    ordered = flat_features.reshape(*ordered_shape)
    inverse_axes = np.argsort([ORDER_AXIS_TO_DIM[name] for name in flatten_order]).tolist()
    return ordered.permute(*inverse_axes).reshape(*train_supercell_shape, unit_cell_atoms, 3)


def block_slices(origin, shape):
    """Return x/y/z slices for one non-periodic training block."""
    return tuple(slice(start, start + size) for start, size in zip(origin, shape))


def differentiable_local_step(predictor, history):
    """Run one differentiable local block step and return next displacement."""
    target_mode = _normalize_target_mode(getattr(predictor, "target_mode", "absolute"))
    block_x_flat = torch.stack(
        [torch_flatten_supercell(frame, predictor.flatten_order) for frame in history],
        dim=0,
    )
    raw = predictor.model(block_x_flat.reshape(1, history.shape[0], -1)).squeeze(0)
    raw_block = torch_unflatten_supercell(
        raw,
        predictor.train_supercell_shape,
        predictor.unit_cell_atoms,
        predictor.flatten_order,
    )
    if target_mode == "delta":
        return history[-1] + raw_block
    if target_mode in {"acceleration", "verlet"}:
        return 2 * history[-1] - history[-2] + raw_block
    return raw_block


def normalized_delta_loss(predicted, target, last_input, epsilon, loss_weights=None):
    """Return normalized delta loss for one block frame."""
    pred_delta = predicted - last_input
    true_delta = target - last_input
    scale = torch.sqrt(torch.mean(true_delta**2)).clamp_min(epsilon)
    return _weighted_mse_loss(pred_delta / scale, true_delta / scale, loss_weights)


def train_crystal_blocks_rollout(
    predictor,
    train_displacements,
    data_len,
    sequence_length,
    rollout_steps,
    rollout_loss_decay,
    delta_loss_weight,
    delta_loss_epsilon,
):
    """Train on local block rollouts sampled from a consecutive trajectory window."""
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if rollout_loss_decay <= 0:
        raise ValueError("rollout_loss_decay must be positive")
    if train_displacements.shape[0] <= sequence_length + rollout_steps:
        raise ValueError("Training window is too short for requested rollout_steps")

    crystal_shape = tuple(int(dim) for dim in train_displacements.shape[1:4])
    origins = _build_supercell_origins(
        crystal_shape,
        predictor.train_supercell_shape,
        stride_shape=(1, 1, 1),
        periodic=False,
    )
    if not origins:
        raise ValueError("No local block origins were generated")

    target_mode = _normalize_target_mode(getattr(predictor, "target_mode", "absolute"))
    delta_loss_weight = _normalize_delta_loss_weight(delta_loss_weight)
    delta_loss_epsilon = _normalize_delta_loss_epsilon(delta_loss_epsilon)
    if target_mode in {"acceleration", "verlet"} and sequence_length < 2:
        raise ValueError(f"target_mode={target_mode!r} requires at least two history frames")

    predictor.train_count = int(data_len * (train_displacements.shape[0] - sequence_length - rollout_steps))
    predictor.train_count = max(1, predictor.train_count)
    steps_per_epoch = max(1, int(np.ceil(predictor.train_count / predictor.batch_size)))
    tensor_displacements = torch.as_tensor(train_displacements, dtype=torch.float32)
    optimizer = optim.Adam(params=predictor.model.parameters(), lr=predictor.lr)
    loss_weights = torch.as_tensor(predictor.loss_weights_supercell(), dtype=torch.float32)
    losses = []
    predictor.model.train()

    for _ in range(predictor.epochs):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            time_start = np.random.randint(0, train_displacements.shape[0] - sequence_length - rollout_steps + 1)
            origin = origins[np.random.randint(0, len(origins))]
            index = block_slices(origin, predictor.train_supercell_shape)
            history_index = (slice(time_start, time_start + sequence_length), *index)
            history = tensor_displacements[history_index].clone()
            loss = torch.zeros((), dtype=torch.float32)
            weight_sum = 0.0

            for step in range(rollout_steps):
                target_index = (time_start + sequence_length + step, *index)
                target = tensor_displacements[target_index]
                predicted = differentiable_local_step(predictor, history)
                step_weight = rollout_loss_decay**step
                loss = loss + step_weight * _weighted_mse_loss(predicted, target, loss_weights)
                if target_mode in {"absolute_delta", "verlet"} and delta_loss_weight > 0:
                    loss = loss + step_weight * delta_loss_weight * normalized_delta_loss(
                        predicted,
                        target,
                        history[-1],
                        delta_loss_epsilon,
                        loss_weights,
                    )
                history = torch.cat([history[1:], predicted.unsqueeze(0)], dim=0)
                weight_sum += step_weight

            loss = loss / weight_sum
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())

        losses.append(epoch_loss / steps_per_epoch)

    predictor.rollout_training = {
        "rollout_steps": rollout_steps,
        "rollout_loss_decay": rollout_loss_decay,
        "delta_loss_weight": delta_loss_weight,
        "delta_loss_epsilon": delta_loss_epsilon,
        "loss_weight_mode": getattr(predictor, "loss_weight_mode", "uniform"),
        "center_loss_weight": getattr(predictor, "center_loss_weight", 1.0),
        "center_loss_alpha": getattr(predictor, "center_loss_alpha", 1.0),
    }
    return losses


def build_default_k_vectors():
    """Build the default reciprocal-space vectors used for S(q,w) scoring."""
    kmin = 2 * np.pi / (NCELLS * LATTICE_PARAMETER)
    kmax = NCELLS * kmin
    kmas = np.zeros((KCOUNT, 3), dtype=np.float32)
    kmas[:, 0] = np.linspace(kmin, kmax, KCOUNT)
    return kmas


def get_sqw_default(coords, dt, step):
    """Calculate S(q,w) with the repository's default k-vector grid."""
    return get_sqw(coords, dt=dt, step=step, kmas=build_default_k_vectors())


def crystal_frames_to_flat_positions(displacements, reference_positions, atom_order):
    """Convert crystal-shaped displacement frames to legacy flat coordinates.

    Parameters
    ----------
    displacements:
        Displacement tensor with shape ``(frames, nx, ny, nz, atoms_per_cell, 3)``.
    reference_positions:
        Equilibrium atom positions in the original flat atom order.
    atom_order:
        Integer tensor that maps each crystal slot to its original flat atom
        index.

    Returns
    -------
    numpy.ndarray
        Absolute positions flattened as ``(frames, atoms * 3)`` for ``get_sqw``.
    """
    positions = reference_positions[atom_order] + displacements
    frames = positions.shape[0]
    atoms = reference_positions.shape[0]
    flat = np.empty((frames, atoms, 3), dtype=np.float32)
    # Restore the original atom order because S(q,w) works with flat trajectories.
    for crystal_index in np.ndindex(atom_order.shape):
        position_index = (slice(None), *crystal_index, slice(None))
        flat[:, atom_order[crystal_index], :] = positions[position_index]
    return flat.reshape(frames, atoms * 3)


def sample_initial_crystal_sequence(displacements, sequence_length, count_steps):
    """Sample an initial crystal displacement history for autoregression."""
    if displacements.shape[0] <= count_steps:
        begin_index = sequence_length
    else:
        begin_index = np.random.randint(low=sequence_length, high=displacements.shape[0] - count_steps)

    return displacements[begin_index - sequence_length : begin_index].astype(np.float32)


def evaluate_model(model, data, reference_displacements, count_steps, count_run, dt, step, periodic):
    """Run inference several times and compare S(q,w) to the training window."""
    if count_run <= 0:
        raise ValueError("count_run must be positive")

    displacements = data["displacements"]
    atom_order = data["atom_order"]
    reference_positions = data["reference_positions"]
    sequence_length = model_sequence_length(data)
    reference_coords = crystal_frames_to_flat_positions(reference_displacements, reference_positions, atom_order)
    xi_ref, yi_ref, jlp_ref = get_sqw_default(reference_coords, dt, step)
    jlp_mean = np.zeros_like(jlp_ref)
    norm = 0.0
    xi_pred = yi_pred = None

    print(f"INFERENCE {count_run} TIMES")
    for _ in range(count_run):
        init = sample_initial_crystal_sequence(displacements, sequence_length, count_steps)
        predicted_displacements = model.run_crystal(count_steps, init, periodic=periodic)
        predicted_coords = crystal_frames_to_flat_positions(predicted_displacements, reference_positions, atom_order)
        xi_pred, yi_pred, jlp_pred = get_sqw_default(predicted_coords, dt, step)

        jlp_mean += jlp_pred
        norm += np.linalg.norm(jlp_pred - jlp_ref)

    norm /= count_run
    jlp_mean /= count_run
    return norm, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean


def model_sequence_length(data):
    """Return the recurrent input history length stored in the training blocks."""
    return int(data["X_blocks"].shape[1])


def plot_sqw(reference_xi, reference_yi, reference_jlp, predicted_xi, predicted_yi, predicted_jlp):
    """Show reference and predicted S(q,w) maps using the same color scheme."""
    import matplotlib.pyplot as plt

    plt.pcolormesh(reference_xi, reference_yi, reference_jlp, cmap="Blues")
    plt.show()
    plt.pcolormesh(predicted_xi, predicted_yi, predicted_jlp, cmap="Blues")
    plt.show()


def save_sqw_plot(reference_xi, reference_yi, reference_jlp, predicted_xi, predicted_yi, predicted_jlp, output_path, title, show=False):
    """Save a side-by-side S(q,w) comparison image."""
    import matplotlib.pyplot as plt

    vmin = float(np.nanmin([np.nanmin(reference_jlp), np.nanmin(predicted_jlp)]))
    vmax = float(np.nanmax([np.nanmax(reference_jlp), np.nanmax(predicted_jlp)]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    mesh = axes[0].pcolormesh(reference_xi, reference_yi, reference_jlp, cmap="Blues", vmin=vmin, vmax=vmax)
    axes[0].set_title("Reference S(q,w)")
    axes[0].set_xlabel("|q|")
    axes[0].set_ylabel("Energy")
    axes[1].pcolormesh(predicted_xi, predicted_yi, predicted_jlp, cmap="Blues", vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted S(q,w)")
    axes[1].set_xlabel("|q|")
    axes[1].set_ylabel("Energy")
    fig.colorbar(mesh, ax=axes, label="Normalized intensity")
    fig.suptitle(title)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"Saved S(q,w) plot to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def save_model(model, models_dir, norm, rnn_type, data_len, count_steps):
    """Persist a selected model with its score and basic training metadata."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    target_mode = getattr(model, "target_mode", "absolute")
    if target_mode == "absolute":
        target_suffix = ""
    elif target_mode == "delta":
        target_suffix = "_delta"
    elif target_mode == "acceleration":
        target_suffix = "_acceleration"
    elif target_mode == "verlet":
        weight = getattr(model, "delta_loss_weight", 1.0)
        epsilon = getattr(model, "delta_loss_epsilon", 1e-6)
        target_suffix = f"_verlet_w{weight:g}_eps{epsilon:g}"
    else:
        weight = getattr(model, "delta_loss_weight", 1.0)
        epsilon = getattr(model, "delta_loss_epsilon", 1e-6)
        target_suffix = f"_absolute_delta_w{weight:g}_eps{epsilon:g}"
    acceleration_weight = getattr(model, "acceleration_loss_weight", 0.0)
    if acceleration_weight > 0:
        acceleration_epsilon = getattr(model, "acceleration_loss_epsilon", 1e-8)
        target_suffix += f"_accw{acceleration_weight:g}_acceps{acceleration_epsilon:g}"
    loss_weight_mode = getattr(model, "loss_weight_mode", "uniform")
    if loss_weight_mode != "uniform":
        center_weight = getattr(model, "center_loss_weight", 1.0)
        target_suffix += f"_loss{loss_weight_mode}_cw{center_weight:g}"
        if loss_weight_mode == "soft_center":
            center_alpha = getattr(model, "center_loss_alpha", 1.0)
            target_suffix += f"_ca{center_alpha:g}"
    if hasattr(model, "rollout_training"):
        rollout = model.rollout_training
        target_suffix += f"_roll{int(rollout['rollout_steps'])}_decay{float(rollout['rollout_loss_decay']):g}"
    temporal_architecture = getattr(model, "temporal_architecture", "stacked")
    if temporal_architecture != "stacked":
        target_suffix += f"_temporal{temporal_architecture.replace('-', '')}"
    filepath = models_path / f"mean_norm_{norm}_{rnn_type.lower()}_crystal{target_suffix}_{int(data_len * count_steps)}.pth"
    torch.save(model, filepath)
    print(f"==============> SAVE MODEL TO FILE - {filepath}")


def main():
    """Train candidate models, evaluate them, and save models below threshold."""
    args = parse_args()
    if args.hidden_size <= 0:
        raise ValueError("hidden-size must be positive")
    if args.num_layers <= 0:
        raise ValueError("num-layers must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0 < args.data_len <= 1:
        raise ValueError("data-len must be in (0, 1]")
    if args.delta_frames <= 0:
        raise ValueError("delta-frames must be positive")
    args.loss_weight_mode = _normalize_loss_weight_mode(args.loss_weight_mode)
    args.center_loss_weight = _normalize_center_loss_weight(args.center_loss_weight)
    args.center_loss_alpha = _normalize_center_loss_alpha(args.center_loss_alpha)
    data = load_training_data(args.data_path)
    sequence_length = model_sequence_length(data)
    if args.temporal_architecture == "frame-layered" and args.num_layers != sequence_length:
        raise ValueError(
            "--temporal-architecture frame-layered requires --num-layers "
            f"to match the data sequence length ({sequence_length})"
        )
    unit_cell_atoms = int(data["X_blocks"].shape[5])
    train_supercell_shape = tuple(int(value) for value in data["train_supercell_shape"])

    for iteration in range(args.count_models):
        print(f"BEGIN ITER = {iteration}")
        predictor = CrystalRNNNet(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            type=args.rnn_type.upper(),
            train_supercell_shape=train_supercell_shape,
            unit_cell_atoms=unit_cell_atoms,
            flatten_order=DEFAULT_FLATTEN_ORDER,
            target_mode=args.target_mode,
            delta_loss_weight=args.delta_loss_weight,
            delta_loss_epsilon=args.delta_loss_epsilon,
            acceleration_loss_weight=args.acceleration_loss_weight,
            acceleration_loss_epsilon=args.acceleration_loss_epsilon,
            loss_weight_mode=args.loss_weight_mode,
            center_loss_weight=args.center_loss_weight,
            center_loss_alpha=args.center_loss_alpha,
            temporal_architecture=args.temporal_architecture,
        )
        predictor.batch_size = args.batch_size
        predictor.epochs = args.epochs
        train_displacements, X_train_blocks, y_train_blocks = sample_train_data(
            data=data,
            delta=args.delta_frames,
            sequence_length=model_sequence_length(data),
        )
        if args.rollout_steps == 1:
            predictor.train_crystal_blocks(X_train_blocks, y_train_blocks, data_len=args.data_len)
        else:
            train_crystal_blocks_rollout(
                predictor=predictor,
                train_displacements=train_displacements,
                data_len=args.data_len,
                sequence_length=model_sequence_length(data),
                rollout_steps=args.rollout_steps,
                rollout_loss_decay=args.rollout_loss_decay,
                delta_loss_weight=args.delta_loss_weight,
                delta_loss_epsilon=args.delta_loss_epsilon,
            )

        norm, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean = evaluate_model(
            model=predictor,
            data=data,
            reference_displacements=train_displacements,
            count_steps=args.count_steps,
            count_run=args.count_run,
            dt=DT,
            step=STEP,
            periodic=args.periodic,
        )
        print("CURRENT_NORM =", norm)

        if args.plot_all:
            output_dir = Path(args.plot_output_dir or args.models_dir)
            save_sqw_plot(
                xi_ref,
                yi_ref,
                jlp_ref,
                xi_pred,
                yi_pred,
                jlp_mean,
                output_dir / f"rnn_iter_{iteration:03d}_sqw.png",
                title=f"RNN iter {iteration}, S(q,w) norm = {norm:.6g}",
                show=args.show_plots,
            )

        if norm < args.save_threshold:
            if not args.plot_all:
                plot_sqw(xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean)
            save_model(predictor, args.models_dir, norm, args.rnn_type, args.data_len, args.count_steps)

    print("DONE ALL JOBS!")


if __name__ == "__main__":
    main()
