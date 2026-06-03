"""Train and select edge-vector temporal RNN crystal acceleration models."""

import argparse
from pathlib import Path

import numpy as np
import torch

from base_classes import (
    CrystalEdgeFinalHiddenRNNNet,
    CrystalEdgeRNNNet,
    CrystalPairEnergyFinalHiddenRNNNet,
    CrystalPairEnergyRNNNet,
    CrystalPairForceFinalHiddenRNNNet,
    CrystalPairForceRNNNet,
)
from find_field_rnn_models import (
    DT,
    STEP,
    crystal_frames_to_flat_positions,
    dynamics_scale_metrics,
    get_sqw_default,
    load_training_data,
    model_sequence_length,
    sample_initial_crystal_sequence,
    sample_train_data,
    save_sqw_plot,
    validate_training_eval_compatibility,
    velocity_distribution_metrics,
)


COUNT_STEPS = 2000
COUNT_RUN = 3
DELTA = 10000
DATA_LEN = 0.2
BATCH_SIZE = 64
HIDDEN_SIZE = 128
RNN_LAYERS = 1
SAVE_THRESHOLD = 3.5
VELOCITY_WINDOW_FRAMES = 10
VELOCITY_HIST_BINS = 80


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count_models", type=int, help="Number of candidate models to train.")
    parser.add_argument("rnn_type", choices=["RNN", "GRU", "LSTM"], help="Temporal recurrent block type.")
    parser.add_argument("--data-path", default="data333.npz")
    parser.add_argument("--eval-data-path", default=None, help="Optional 300 K evaluation dataset.")
    parser.add_argument("--models-dir", default="models333_edge_rnn")
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--count-steps", type=int, default=COUNT_STEPS)
    parser.add_argument("--count-run", type=int, default=COUNT_RUN)
    parser.add_argument("--delta-frames", type=int, default=DELTA)
    parser.add_argument("--data-len", type=float, default=DATA_LEN)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--rnn-layers", type=int, default=RNN_LAYERS)
    parser.add_argument(
        "--rnn-readout-mode",
        choices=["last-output", "final-hidden"],
        default="last-output",
        help=(
            "How to reduce the recurrent sequence. 'last-output' is the current edge-RNN behavior; "
            "'final-hidden' uses h_n from the last bidirectional layer, matching the original flat RNN readout."
        ),
    )
    parser.add_argument(
        "--temporal-architecture",
        choices=["stacked", "frame-layered"],
        default="stacked",
        help=(
            "Temporal core for pair-energy models. 'stacked' uses PyTorch "
            "nn.RNN/GRU/LSTM num_layers; 'frame-layered' assigns each history "
            "frame to its own recurrent cell, so rnn-layers must equal the "
            "data sequence length."
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=["edge", "pair-force", "pair-energy"],
        default="edge",
        help=(
            "Model output parameterization. 'edge' predicts central accelerations directly; "
            "'pair-force' sums antisymmetric pair contributions and can scatter them during rollout; "
            "'pair-energy' differentiates an even scalar pair energy to obtain conservative pair forces."
        ),
    )
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neighbor-shells", type=int, default=2)
    parser.add_argument("--cutoff-scale", type=float, default=1.05)
    parser.add_argument("--acceleration-normalization", choices=["none", "global", "channel"], default="channel")
    parser.add_argument(
        "--training-target",
        choices=["displacement", "force"],
        default="displacement",
        help=(
            "Acceleration target source. 'displacement' uses u[n+1]-2u[n]+u[n-1]; "
            "'force' uses force_acceleration_blocks imported from fx fy fz."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-threshold", type=float, default=SAVE_THRESHOLD)
    parser.add_argument("--save-all", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot-all", action="store_true")
    parser.add_argument("--plot-output-dir", default=None)
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--velocity-window-frames", type=int, default=VELOCITY_WINDOW_FRAMES)
    parser.add_argument("--velocity-hist-bins", type=int, default=VELOCITY_HIST_BINS)
    parser.add_argument("--velocity-score-weight", type=float, default=0.0)
    parser.add_argument(
        "--acceleration-score-weight",
        type=float,
        default=0.0,
        help=(
            "Selection-score weight for rollout acceleration RMS scale mismatch. "
            "The score uses abs(log(acceleration_rms_ratio)) and the same end-window term."
        ),
    )
    parser.add_argument(
        "--displacement-moment-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary one-step loss that matches predicted/reference displacement "
            "moments per Cartesian component. Zero disables it."
        ),
    )
    parser.add_argument(
        "--displacement-moment-mean-weight",
        type=float,
        default=1.0,
        help="Relative weight of the per-axis displacement mean term inside the moment loss.",
    )
    parser.add_argument(
        "--displacement-moment-std-weight",
        type=float,
        default=1.0,
        help="Relative weight of the per-axis displacement standard-deviation term inside the moment loss.",
    )
    parser.add_argument(
        "--displacement-moment-rms-weight",
        type=float,
        default=0.0,
        help="Relative weight of the per-axis displacement RMS term inside the moment loss.",
    )
    parser.add_argument(
        "--displacement-moment-component-weights",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("WX", "WY", "WZ"),
        help=(
            "Per-component x/y/z weights inside the displacement moment loss. "
            "Weights are normalized by their mean, so 1 2 2 emphasizes y/z without changing the average scale."
        ),
    )
    parser.add_argument(
        "--displacement-moment-loss-epsilon",
        type=float,
        default=1e-12,
        help="Numerical stabilizer for displacement moment ratios.",
    )
    parser.add_argument(
        "--power-mean-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary loss on the mini-batch mean power bias "
            "mean(a_pred dot v_ref) - mean(a_ref dot v_ref). Zero disables it."
        ),
    )
    parser.add_argument(
        "--power-mean-loss-epsilon",
        type=float,
        default=1e-12,
        help="Numerical stabilizer for the normalized mean-power loss.",
    )
    parser.add_argument(
        "--q-power-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary loss on q-shell-resolved acceleration power. "
            "Zero disables it."
        ),
    )
    parser.add_argument(
        "--q-power-loss-mode",
        choices=["match", "positive-excess"],
        default="positive-excess",
        help=(
            "'match' fits reference shell powers; 'positive-excess' only "
            "penalizes predicted shell power above reference plus margin."
        ),
    )
    parser.add_argument(
        "--q-power-loss-sample-count",
        type=int,
        default=2,
        help="Number of full periodic blocks used when q-power loss is evaluated.",
    )
    parser.add_argument(
        "--q-power-loss-interval",
        type=int,
        default=10,
        help="Evaluate q-power loss once every N mini-batches.",
    )
    parser.add_argument(
        "--q-power-loss-margin",
        type=float,
        default=0.0,
        help="Dimensionless positive-excess margin in reference-shell RMS units.",
    )
    parser.add_argument(
        "--q-power-loss-epsilon",
        type=float,
        default=1e-12,
        help="Numerical stabilizer for the q-power shell loss.",
    )
    parser.add_argument(
        "--q-power-loss-exclude-q-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude the spatial q=0 shell from q-power loss.",
    )
    parser.add_argument(
        "--acceleration-rms-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary one-step loss on log RMS ratio of predicted/reference accelerations. "
            "Zero disables it."
        ),
    )
    parser.add_argument(
        "--acceleration-batch-rms-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary loss on per-Cartesian-component acceleration RMS over each mini-batch. "
            "Zero disables it."
        ),
    )
    parser.add_argument(
        "--acceleration-tail-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary loss on normalized fourth moments of acceleration components over each mini-batch. "
            "Zero disables it."
        ),
    )
    parser.add_argument(
        "--acceleration-over-rms-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Asymmetric RMS-ratio loss weight for overestimated accelerations: "
            "mean(ReLU(rms_pred / rms_ref - 1)^2). Zero disables this side."
        ),
    )
    parser.add_argument(
        "--acceleration-under-rms-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Asymmetric RMS-ratio loss weight for underestimated accelerations: "
            "mean(ReLU(1 - rms_pred / rms_ref)^2). Zero disables this side."
        ),
    )
    parser.add_argument(
        "--acceleration-over-rms-loss-power",
        type=float,
        default=2.0,
        help="Power used in the overestimated-acceleration RMS-ratio loss.",
    )
    parser.add_argument(
        "--acceleration-under-rms-loss-power",
        type=float,
        default=2.0,
        help="Power used in the underestimated-acceleration RMS-ratio loss.",
    )
    parser.add_argument(
        "--rms-loss-epsilon",
        type=float,
        default=1e-12,
        help="Numerical floor for RMS-ratio auxiliary losses.",
    )
    parser.add_argument(
        "--curl-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary loss on the antisymmetric part of the local acceleration Jacobian "
            "d a_center / d u_center. Zero disables it."
        ),
    )
    parser.add_argument(
        "--curl-loss-sample-count",
        type=int,
        default=4,
        help="Number of samples per mini-batch used for the higher-order curl loss.",
    )
    parser.add_argument(
        "--curl-loss-interval",
        type=int,
        default=1,
        help="Apply the higher-order curl loss once every N mini-batches.",
    )
    parser.add_argument(
        "--curl-loss-epsilon",
        type=float,
        default=1e-12,
        help="Numerical floor for normalized curl loss.",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def validate_args(args):
    """Validate scalar command-line parameters."""
    if args.count_models <= 0:
        raise ValueError("count_models must be positive")
    if args.count_steps <= 0:
        raise ValueError("count-steps must be positive")
    if args.count_run <= 0:
        raise ValueError("count-run must be positive")
    if args.delta_frames <= 0:
        raise ValueError("delta-frames must be positive")
    if not 0 < args.data_len <= 1:
        raise ValueError("data-len must be in (0, 1]")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.hidden_size <= 0:
        raise ValueError("hidden-size must be positive")
    if args.rnn_layers <= 0:
        raise ValueError("rnn-layers must be positive")
    if args.neighbor_shells != 2:
        raise ValueError("this first edge-RNN implementation supports --neighbor-shells 2")
    if args.cutoff_scale <= 0:
        raise ValueError("cutoff-scale must be positive")
    if args.velocity_window_frames <= 0:
        raise ValueError("velocity-window-frames must be positive")
    if args.velocity_hist_bins <= 0:
        raise ValueError("velocity-hist-bins must be positive")
    if args.velocity_score_weight < 0:
        raise ValueError("velocity-score-weight must be non-negative")
    if args.acceleration_score_weight < 0:
        raise ValueError("acceleration-score-weight must be non-negative")
    if args.displacement_moment_loss_weight < 0:
        raise ValueError("displacement-moment-loss-weight must be non-negative")
    if args.displacement_moment_mean_weight < 0:
        raise ValueError("displacement-moment-mean-weight must be non-negative")
    if args.displacement_moment_std_weight < 0:
        raise ValueError("displacement-moment-std-weight must be non-negative")
    if args.displacement_moment_rms_weight < 0:
        raise ValueError("displacement-moment-rms-weight must be non-negative")
    if any(weight < 0 for weight in args.displacement_moment_component_weights):
        raise ValueError("displacement-moment-component-weights must be non-negative")
    if not any(weight > 0 for weight in args.displacement_moment_component_weights):
        raise ValueError("At least one displacement moment component weight must be positive")
    if args.displacement_moment_loss_epsilon <= 0:
        raise ValueError("displacement-moment-loss-epsilon must be positive")
    if args.power_mean_loss_weight < 0:
        raise ValueError("power-mean-loss-weight must be non-negative")
    if args.power_mean_loss_epsilon <= 0:
        raise ValueError("power-mean-loss-epsilon must be positive")
    if args.q_power_loss_weight < 0:
        raise ValueError("q-power-loss-weight must be non-negative")
    if args.q_power_loss_sample_count <= 0:
        raise ValueError("q-power-loss-sample-count must be positive")
    if args.q_power_loss_interval <= 0:
        raise ValueError("q-power-loss-interval must be positive")
    if args.q_power_loss_margin < 0:
        raise ValueError("q-power-loss-margin must be non-negative")
    if args.q_power_loss_epsilon <= 0:
        raise ValueError("q-power-loss-epsilon must be positive")
    if args.acceleration_rms_loss_weight < 0:
        raise ValueError("acceleration-rms-loss-weight must be non-negative")
    if args.acceleration_batch_rms_loss_weight < 0:
        raise ValueError("acceleration-batch-rms-loss-weight must be non-negative")
    if args.acceleration_tail_loss_weight < 0:
        raise ValueError("acceleration-tail-loss-weight must be non-negative")
    if args.acceleration_over_rms_loss_weight < 0:
        raise ValueError("acceleration-over-rms-loss-weight must be non-negative")
    if args.acceleration_under_rms_loss_weight < 0:
        raise ValueError("acceleration-under-rms-loss-weight must be non-negative")
    if args.acceleration_over_rms_loss_power <= 0:
        raise ValueError("acceleration-over-rms-loss-power must be positive")
    if args.acceleration_under_rms_loss_power <= 0:
        raise ValueError("acceleration-under-rms-loss-power must be positive")
    if args.rms_loss_epsilon <= 0:
        raise ValueError("rms-loss-epsilon must be positive")
    if args.curl_loss_weight < 0:
        raise ValueError("curl-loss-weight must be non-negative")
    if args.curl_loss_sample_count <= 0:
        raise ValueError("curl-loss-sample-count must be positive")
    if args.curl_loss_interval <= 0:
        raise ValueError("curl-loss-interval must be positive")
    if args.curl_loss_epsilon <= 0:
        raise ValueError("curl-loss-epsilon must be positive")
    if (
        args.displacement_moment_loss_weight > 0
        and args.displacement_moment_mean_weight == 0
        and args.displacement_moment_std_weight == 0
        and args.displacement_moment_rms_weight == 0
    ):
        raise ValueError("At least one displacement moment sub-weight must be positive")


def evaluate_model(model, data, reference_displacements, count_steps, count_run, rng, velocity_window_frames, velocity_hist_bins):
    """Evaluate an edge-RNN model by rollout S(q,w) and dynamics scale metrics."""
    displacements = data["displacements"]
    atom_order = data["atom_order"]
    reference_positions = data["reference_positions"]
    sequence_length = model_sequence_length(data)
    reference_coords = crystal_frames_to_flat_positions(reference_displacements, reference_positions, atom_order)
    xi_ref, yi_ref, jlp_ref = get_sqw_default(reference_coords, DT, STEP)
    jlp_mean = np.zeros_like(jlp_ref)
    norm = 0.0
    velocity_metrics_sum = None
    scale_metrics_sum = None
    xi_pred = yi_pred = None

    print(f"INFERENCE {count_run} TIMES")
    for _ in range(count_run):
        init = sample_initial_crystal_sequence(displacements, sequence_length, count_steps, rng)
        predicted_displacements = model.run_crystal(count_steps, init, periodic=True)
        predicted_coords = crystal_frames_to_flat_positions(predicted_displacements, reference_positions, atom_order)
        xi_pred, yi_pred, jlp_pred = get_sqw_default(predicted_coords, DT, STEP)
        jlp_mean += jlp_pred
        norm += np.linalg.norm(jlp_pred - jlp_ref)
        velocity_metrics = velocity_distribution_metrics(
            predicted_displacements=predicted_displacements,
            reference_displacements=reference_displacements,
            dt=DT,
            window_frames=velocity_window_frames,
            bins=velocity_hist_bins,
        )
        scale_metrics = dynamics_scale_metrics(
            predicted_displacements=predicted_displacements,
            reference_displacements=reference_displacements,
            window_frames=velocity_window_frames,
        )
        if velocity_metrics_sum is None:
            velocity_metrics_sum = {key: 0.0 for key in velocity_metrics}
        if scale_metrics_sum is None:
            scale_metrics_sum = {key: 0.0 for key in scale_metrics}
        for key, value in velocity_metrics.items():
            velocity_metrics_sum[key] += float(value)
        for key, value in scale_metrics.items():
            scale_metrics_sum[key] += float(value)

    norm /= count_run
    jlp_mean /= count_run
    metrics = {key: value / count_run for key, value in velocity_metrics_sum.items()}
    metrics.update({key: value / count_run for key, value in scale_metrics_sum.items()})
    return norm, metrics, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean


def sample_edge_train_data(data, delta, sequence_length, rng, training_target, label="TRAIN"):
    """Select a random consecutive frame window and edge-RNN target blocks."""
    displacements = data["displacements"]
    X_blocks = data["X_blocks"]
    target_key = "force_acceleration_blocks" if training_target == "force" else "y_blocks"
    if target_key not in data:
        raise ValueError(
            f"{target_key} is required for --training-target {training_target!r}. "
            "Regenerate the .npz with prepare_crystal_data.py --include-forces."
        )
    target_blocks = data[target_key]
    frame_count = displacements.shape[0]
    total_time_windows = frame_count - sequence_length
    if total_time_windows <= 0:
        raise ValueError("Need more displacement frames than sequence_length")
    if X_blocks.shape[0] != target_blocks.shape[0]:
        raise ValueError(f"X_blocks and {target_key} must contain the same number of samples")
    if X_blocks.shape[0] % total_time_windows != 0:
        raise ValueError("X_blocks count is inconsistent with displacements and sequence_length")

    blocks_per_time_window = X_blocks.shape[0] // total_time_windows
    if frame_count <= delta:
        start_frame = 0
        window_frame_count = frame_count
    else:
        start_frame = int(rng.integers(0, frame_count - delta + 1))
        window_frame_count = int(delta)

    start_sample = start_frame * blocks_per_time_window
    sample_count = (window_frame_count - sequence_length) * blocks_per_time_window
    if sample_count <= 0:
        raise ValueError("Selected training window is too short for sequence_length")

    stop_sample = start_sample + sample_count
    print(f"{label} FRAMES {start_frame}:{start_frame + window_frame_count}")
    print(f"{label} BLOCKS {start_sample}:{stop_sample}")
    print(f"{label} TARGET {target_key}")
    return (
        displacements[start_frame : start_frame + window_frame_count],
        X_blocks[start_sample:stop_sample],
        target_blocks[start_sample:stop_sample],
        data["y_blocks"][start_sample:stop_sample],
    )


def weight_tag(value):
    """Return a compact stable tag for filename weights."""
    return f"{float(value):g}"


def acceleration_scale_score(metrics, epsilon=1e-12):
    """Return a symmetric rollout acceleration-scale mismatch score."""
    rms_ratio = max(float(metrics["acceleration_rms_ratio"]), float(epsilon))
    end_rms_ratio = max(float(metrics["acceleration_end_rms_ratio"]), float(epsilon))
    return 0.5 * (abs(np.log(rms_ratio)) + abs(np.log(end_rms_ratio)))


def save_model(model, models_dir, norm, args):
    """Save a trained edge-RNN model."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    training_target = getattr(model, "training_target", args.training_target)
    moment_tag = ""
    if getattr(model, "displacement_moment_loss_weight", 0.0) > 0:
        moment_tag = (
            f"_dmom{weight_tag(model.displacement_moment_loss_weight)}"
            f"m{weight_tag(model.displacement_moment_mean_weight)}"
            f"s{weight_tag(model.displacement_moment_std_weight)}"
            f"r{weight_tag(model.displacement_moment_rms_weight)}"
            f"c{'-'.join(weight_tag(value) for value in model.displacement_moment_component_weights)}"
        )
    power_tag = ""
    if getattr(model, "power_mean_loss_weight", 0.0) > 0:
        power_tag = f"_pmean{weight_tag(model.power_mean_loss_weight)}"
    q_power_tag = ""
    if getattr(model, "q_power_loss_weight", 0.0) > 0:
        q_power_mode = str(getattr(model, "q_power_loss_mode", "positive-excess")).replace("-", "")
        q_power_tag = (
            f"_qpow{weight_tag(model.q_power_loss_weight)}"
            f"_{q_power_mode}"
            f"_qs{int(getattr(model, 'q_power_loss_sample_count', 2))}"
            f"_qi{int(getattr(model, 'q_power_loss_interval', 10))}"
        )
    acceleration_rms_tag = ""
    if getattr(model, "acceleration_rms_loss_weight", 0.0) > 0:
        acceleration_rms_tag = f"_arms{weight_tag(model.acceleration_rms_loss_weight)}"
    acceleration_batch_rms_tag = ""
    if getattr(model, "acceleration_batch_rms_loss_weight", 0.0) > 0:
        acceleration_batch_rms_tag = f"_abrms{weight_tag(model.acceleration_batch_rms_loss_weight)}"
    acceleration_tail_tag = ""
    if getattr(model, "acceleration_tail_loss_weight", 0.0) > 0:
        acceleration_tail_tag = f"_atail{weight_tag(model.acceleration_tail_loss_weight)}"
    acceleration_asymmetric_rms_tag = ""
    if (
        getattr(model, "acceleration_over_rms_loss_weight", 0.0) > 0
        or getattr(model, "acceleration_under_rms_loss_weight", 0.0) > 0
    ):
        over_power = getattr(model, "acceleration_over_rms_loss_power", 2.0)
        under_power = getattr(model, "acceleration_under_rms_loss_power", 2.0)
        acceleration_asymmetric_rms_tag = (
            f"_aover{weight_tag(model.acceleration_over_rms_loss_weight)}"
            f"_aunder{weight_tag(model.acceleration_under_rms_loss_weight)}"
            f"_op{weight_tag(over_power)}"
            f"_up{weight_tag(under_power)}"
        )
    curl_tag = ""
    if getattr(model, "curl_loss_weight", 0.0) > 0:
        curl_tag = (
            f"_curl{weight_tag(model.curl_loss_weight)}"
            f"_cs{int(model.curl_loss_sample_count)}"
            f"_ci{int(model.curl_loss_interval)}"
    )
    architecture = getattr(model, "architecture", getattr(args, "architecture", "edge"))
    architecture_tag = {
        "edge": "edge_rnn",
        "pair-force": "pair_force_rnn",
        "pair-energy": "pair_energy_rnn",
    }.get(architecture, f"{architecture.replace('-', '_')}_rnn")
    readout_mode = getattr(model, "rnn_readout_mode", getattr(args, "rnn_readout_mode", "last-output"))
    readout_tag = "" if readout_mode == "last-output" else f"_readout{readout_mode.replace('-', '')}"
    temporal_architecture = getattr(model, "temporal_architecture", getattr(args, "temporal_architecture", "stacked"))
    temporal_tag = "" if temporal_architecture == "stacked" else f"_temporal{temporal_architecture.replace('-', '')}"
    filename = (
        f"mean_norm_{norm}_{args.rnn_type.lower()}_{architecture_tag}_acceleration"
        f"_h{model.hidden_size}_rl{model.rnn_layers}"
        f"{readout_tag}"
        f"{temporal_tag}"
        f"{'_bidir' if model.bidirectional else ''}"
        f"_shells{model.neighbor_shells}_n{model.neighbor_count}"
        f"_target{training_target}"
        f"_accnorm{model.acceleration_normalization}"
        f"{moment_tag}{power_tag}{q_power_tag}{acceleration_rms_tag}{acceleration_batch_rms_tag}"
        f"{acceleration_tail_tag}{acceleration_asymmetric_rms_tag}{curl_tag}.pth"
    )
    path = models_path / filename
    torch.save(model, path)
    print(f"==============> SAVE MODEL TO FILE - {path}")
    return path


def write_metrics(path, rows):
    """Write model-search metrics to TSV."""
    fields = [
        "iteration",
        "model_path",
        "data_path",
        "eval_data_path",
        "sqw_norm",
        "selection_score",
        "velocity_score",
        "acceleration_score",
        "velocity_end_speed_ratio",
        "velocity_rms_ratio",
        "velocity_end_rms_ratio",
        "acceleration_rms_ratio",
        "acceleration_end_rms_ratio",
        "rnn_type",
        "hidden_size",
        "rnn_layers",
        "rnn_readout_mode",
        "temporal_architecture",
        "architecture",
        "bidirectional",
        "neighbor_shells",
        "neighbor_count",
        "cutoff_scale",
        "lattice_parameter",
        "acceleration_normalization",
        "training_target",
        "delta_frames",
        "data_len",
        "count_steps",
        "count_run",
        "epochs",
        "batch_size",
        "learning_rate",
        "displacement_moment_loss_weight",
        "displacement_moment_mean_weight",
        "displacement_moment_std_weight",
        "displacement_moment_rms_weight",
        "displacement_moment_component_weights",
        "displacement_moment_loss_epsilon",
        "power_mean_loss_weight",
        "power_mean_loss_epsilon",
        "q_power_loss_weight",
        "q_power_loss_mode",
        "q_power_loss_sample_count",
        "q_power_loss_interval",
        "q_power_loss_margin",
        "q_power_loss_epsilon",
        "q_power_loss_exclude_q_zero",
        "acceleration_rms_loss_weight",
        "acceleration_batch_rms_loss_weight",
        "acceleration_tail_loss_weight",
        "acceleration_over_rms_loss_weight",
        "acceleration_under_rms_loss_weight",
        "acceleration_over_rms_loss_power",
        "acceleration_under_rms_loss_power",
        "rms_loss_epsilon",
        "curl_loss_weight",
        "curl_loss_sample_count",
        "curl_loss_interval",
        "curl_loss_epsilon",
        "device",
        "final_train_loss",
        "best_train_loss",
    ]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append(
            "\t".join(
                str(row[field]) if isinstance(row[field], str) else f"{row[field]:.10g}"
                for field in fields
            )
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """Train and evaluate edge-vector RNN candidates."""
    args = parse_args()
    validate_args(args)
    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    data = load_training_data(args.data_path)
    eval_data = data if args.eval_data_path is None else load_training_data(args.eval_data_path)
    validate_training_eval_compatibility(data, eval_data)
    sequence_length = model_sequence_length(data)
    if args.temporal_architecture == "frame-layered":
        if args.architecture != "pair-energy":
            raise ValueError("--temporal-architecture frame-layered is currently implemented only for pair-energy")
        if args.rnn_layers != sequence_length:
            raise ValueError(
                "--temporal-architecture frame-layered requires --rnn-layers "
                f"to match the data sequence length ({sequence_length})"
            )
    rows = []

    for iteration in range(args.count_models):
        print(f"BEGIN ITER = {iteration}")
        if args.architecture == "pair-energy" and args.rnn_readout_mode == "final-hidden":
            model_cls = CrystalPairEnergyFinalHiddenRNNNet
        elif args.architecture == "pair-energy":
            model_cls = CrystalPairEnergyRNNNet
        elif args.architecture == "pair-force" and args.rnn_readout_mode == "final-hidden":
            model_cls = CrystalPairForceFinalHiddenRNNNet
        elif args.architecture == "pair-force":
            model_cls = CrystalPairForceRNNNet
        elif args.rnn_readout_mode == "final-hidden":
            model_cls = CrystalEdgeFinalHiddenRNNNet
        else:
            model_cls = CrystalEdgeRNNNet
        model_kwargs = {
            "reference_positions": data["reference_positions"],
            "atom_order": data["atom_order"],
            "box_lengths": data["box_lengths"],
            "hidden_size": args.hidden_size,
            "rnn_layers": args.rnn_layers,
            "type": args.rnn_type,
            "bidirectional": args.bidirectional,
            "neighbor_shells": args.neighbor_shells,
            "cutoff_scale": args.cutoff_scale,
            "acceleration_normalization": args.acceleration_normalization,
            "rnn_readout_mode": args.rnn_readout_mode,
            "device": args.device,
        }
        if args.architecture == "pair-energy":
            model_kwargs["temporal_architecture"] = args.temporal_architecture
        model = model_cls(**model_kwargs)
        print("DEVICE =", model.torch_device)
        print("ARCHITECTURE =", args.architecture)
        print("RNN_READOUT_MODE =", model.rnn_readout_mode)
        print("TEMPORAL_ARCHITECTURE =", getattr(model, "temporal_architecture", "stacked"))
        print("NEIGHBOR_COUNT =", model.neighbor_count)
        model.batch_size = args.batch_size
        model.epochs = args.epochs
        model.lr = args.learning_rate
        model.displacement_moment_loss_weight = args.displacement_moment_loss_weight
        model.displacement_moment_mean_weight = args.displacement_moment_mean_weight
        model.displacement_moment_std_weight = args.displacement_moment_std_weight
        model.displacement_moment_rms_weight = args.displacement_moment_rms_weight
        model.displacement_moment_component_weights = np.asarray(
            args.displacement_moment_component_weights,
            dtype=np.float32,
        )
        model.displacement_moment_loss_epsilon = args.displacement_moment_loss_epsilon
        model.power_mean_loss_weight = args.power_mean_loss_weight
        model.power_mean_loss_epsilon = args.power_mean_loss_epsilon
        model.q_power_loss_weight = args.q_power_loss_weight
        model.q_power_loss_mode = args.q_power_loss_mode
        model.q_power_loss_sample_count = args.q_power_loss_sample_count
        model.q_power_loss_interval = args.q_power_loss_interval
        model.q_power_loss_margin = args.q_power_loss_margin
        model.q_power_loss_epsilon = args.q_power_loss_epsilon
        model.q_power_loss_exclude_q_zero = args.q_power_loss_exclude_q_zero
        model.acceleration_rms_loss_weight = args.acceleration_rms_loss_weight
        model.acceleration_batch_rms_loss_weight = args.acceleration_batch_rms_loss_weight
        model.acceleration_tail_loss_weight = args.acceleration_tail_loss_weight
        model.acceleration_over_rms_loss_weight = args.acceleration_over_rms_loss_weight
        model.acceleration_under_rms_loss_weight = args.acceleration_under_rms_loss_weight
        model.acceleration_over_rms_loss_power = args.acceleration_over_rms_loss_power
        model.acceleration_under_rms_loss_power = args.acceleration_under_rms_loss_power
        model.rms_loss_epsilon = args.rms_loss_epsilon
        model.curl_loss_weight = args.curl_loss_weight
        model.curl_loss_sample_count = args.curl_loss_sample_count
        model.curl_loss_interval = args.curl_loss_interval
        model.curl_loss_epsilon = args.curl_loss_epsilon

        train_displacements, X_train_blocks, target_train_blocks, displacement_y_train_blocks = sample_edge_train_data(
            data=data,
            delta=args.delta_frames,
            sequence_length=model_sequence_length(data),
            rng=rng,
            training_target=args.training_target,
            label="TRAIN",
        )
        losses = model.train_crystal_blocks(
            X_train_blocks,
            target_train_blocks,
            data_len=args.data_len,
            training_target=args.training_target,
            displacement_y_blocks=displacement_y_train_blocks,
        )
        if eval_data is data:
            eval_reference_displacements = train_displacements
        else:
            eval_reference_displacements, _, _ = sample_train_data(
                data=eval_data,
                delta=args.delta_frames,
                sequence_length=model_sequence_length(eval_data),
                rng=rng,
                label="EVAL",
            )

        norm, metrics, xi_ref, yi_ref, jlp_ref, xi_pred, yi_pred, jlp_mean = evaluate_model(
            model=model,
            data=eval_data,
            reference_displacements=eval_reference_displacements,
            count_steps=args.count_steps,
            count_run=args.count_run,
            rng=rng,
            velocity_window_frames=args.velocity_window_frames,
            velocity_hist_bins=args.velocity_hist_bins,
        )
        acceleration_score = acceleration_scale_score(metrics)
        selection_score = (
            norm
            + args.velocity_score_weight * metrics["velocity_score"]
            + args.acceleration_score_weight * acceleration_score
        )
        print("CURRENT_NORM =", norm)
        print("VELOCITY_SCORE =", metrics["velocity_score"])
        print("ACCELERATION_SCORE =", acceleration_score)
        print("VELOCITY_END_SPEED_RATIO =", metrics["velocity_end_speed_ratio"])
        print("ACCELERATION_RMS_RATIO =", metrics["acceleration_rms_ratio"])
        print("ACCELERATION_END_RMS_RATIO =", metrics["acceleration_end_rms_ratio"])
        print("VELOCITY_RMS_RATIO =", metrics["velocity_rms_ratio"])
        print("VELOCITY_END_RMS_RATIO =", metrics["velocity_end_rms_ratio"])
        print("SELECTION_SCORE =", selection_score)

        if args.plot_all:
            output_dir = Path(args.plot_output_dir or args.models_dir)
            save_sqw_plot(
                xi_ref,
                yi_ref,
                jlp_ref,
                xi_pred,
                yi_pred,
                jlp_mean,
                output_dir / f"{args.architecture.replace('-', '_')}_iter_{iteration:03d}_sqw.png",
                title=f"{args.architecture} iter {iteration}, S(q,w) norm = {norm:.6g}",
                show=args.show_plots,
            )

        model_path = ""
        if args.save_all or selection_score < args.save_threshold:
            model_path = str(save_model(model, args.models_dir, norm, args))

        rows.append(
            {
                "iteration": iteration,
                "model_path": model_path,
                "data_path": args.data_path,
                "eval_data_path": args.eval_data_path or args.data_path,
                "sqw_norm": norm,
                "selection_score": selection_score,
                "velocity_score": metrics["velocity_score"],
                "acceleration_score": acceleration_score,
                "velocity_end_speed_ratio": metrics["velocity_end_speed_ratio"],
                "velocity_rms_ratio": metrics["velocity_rms_ratio"],
                "velocity_end_rms_ratio": metrics["velocity_end_rms_ratio"],
                "acceleration_rms_ratio": metrics["acceleration_rms_ratio"],
                "acceleration_end_rms_ratio": metrics["acceleration_end_rms_ratio"],
                "rnn_type": args.rnn_type,
                "hidden_size": model.hidden_size,
                "rnn_layers": model.rnn_layers,
                "rnn_readout_mode": model.rnn_readout_mode,
                "temporal_architecture": getattr(model, "temporal_architecture", "stacked"),
                "architecture": args.architecture,
                "bidirectional": bool(model.bidirectional),
                "neighbor_shells": model.neighbor_shells,
                "neighbor_count": model.neighbor_count,
                "cutoff_scale": args.cutoff_scale,
                "lattice_parameter": model.lattice_parameter,
                "acceleration_normalization": model.acceleration_normalization,
                "training_target": model.training_target,
                "delta_frames": args.delta_frames,
                "data_len": args.data_len,
                "count_steps": args.count_steps,
                "count_run": args.count_run,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "displacement_moment_loss_weight": model.displacement_moment_loss_weight,
                "displacement_moment_mean_weight": model.displacement_moment_mean_weight,
                "displacement_moment_std_weight": model.displacement_moment_std_weight,
                "displacement_moment_rms_weight": model.displacement_moment_rms_weight,
                "displacement_moment_component_weights": ",".join(
                    f"{value:g}" for value in model.displacement_moment_component_weights
                ),
                "displacement_moment_loss_epsilon": model.displacement_moment_loss_epsilon,
                "power_mean_loss_weight": model.power_mean_loss_weight,
                "power_mean_loss_epsilon": model.power_mean_loss_epsilon,
                "q_power_loss_weight": model.q_power_loss_weight,
                "q_power_loss_mode": model.q_power_loss_mode,
                "q_power_loss_sample_count": model.q_power_loss_sample_count,
                "q_power_loss_interval": model.q_power_loss_interval,
                "q_power_loss_margin": model.q_power_loss_margin,
                "q_power_loss_epsilon": model.q_power_loss_epsilon,
                "q_power_loss_exclude_q_zero": bool(model.q_power_loss_exclude_q_zero),
                "acceleration_rms_loss_weight": model.acceleration_rms_loss_weight,
                "acceleration_batch_rms_loss_weight": model.acceleration_batch_rms_loss_weight,
                "acceleration_tail_loss_weight": model.acceleration_tail_loss_weight,
                "acceleration_over_rms_loss_weight": model.acceleration_over_rms_loss_weight,
                "acceleration_under_rms_loss_weight": model.acceleration_under_rms_loss_weight,
                "acceleration_over_rms_loss_power": model.acceleration_over_rms_loss_power,
                "acceleration_under_rms_loss_power": model.acceleration_under_rms_loss_power,
                "rms_loss_epsilon": model.rms_loss_epsilon,
                "curl_loss_weight": model.curl_loss_weight,
                "curl_loss_sample_count": model.curl_loss_sample_count,
                "curl_loss_interval": model.curl_loss_interval,
                "curl_loss_epsilon": model.curl_loss_epsilon,
                "device": str(model.torch_device),
                "final_train_loss": float(losses[-1]) if losses else np.nan,
                "best_train_loss": float(np.min(losses)) if losses else np.nan,
            }
        )
        if args.metrics_path:
            write_metrics(args.metrics_path, rows)

    print("DONE ALL JOBS!")


if __name__ == "__main__":
    main()
