"""Plot velocity histograms for predicted and reference inference trajectories."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DT = 0.02


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare velocity distributions in the first and last time windows "
            "of one or more inference .npz files."
        )
    )
    parser.add_argument("--input-path", action="append", required=True, help="Inference output .npz file.")
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional label for the matching --input-path. Repeat once per input.",
    )
    parser.add_argument("--output-path", default="velocity_histograms.png")
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument(
        "--window-frames",
        type=int,
        default=10,
        help="Number of velocity intervals from the beginning and end of the trajectory.",
    )
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Use symmetric component histogram limits from this absolute-value percentile.",
    )
    parser.add_argument("--speed-percentile", type=float, default=99.5)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_displacements(data, predicted):
    """Load predicted or reference displacement frames from an inference result."""
    key = "predicted_displacements" if predicted else "reference_displacements"
    if key in data.files:
        return np.asarray(data[key], dtype=np.float64)
    raise ValueError(f"Missing {key!r}; this script expects displacement-based inference outputs")


def velocity_windows(displacements, dt, window_frames):
    """Return beginning and ending velocity windows from displacement frames."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if window_frames <= 0:
        raise ValueError("window-frames must be positive")
    if displacements.shape[0] <= window_frames:
        raise ValueError("trajectory is too short for the requested velocity window")

    velocities = np.diff(displacements, axis=0) / dt
    return velocities[:window_frames], velocities[-window_frames:]


def flatten_components(velocities):
    """Return flattened velocity components."""
    return np.asarray(velocities, dtype=np.float64).reshape(-1)


def flatten_speeds(velocities):
    """Return flattened velocity magnitudes."""
    velocities = np.asarray(velocities, dtype=np.float64)
    return np.linalg.norm(velocities.reshape(-1, 3), axis=1)


def robust_component_limit(*arrays, percentile):
    """Return a symmetric robust limit for component histograms."""
    values = np.concatenate([np.abs(flatten_components(array)) for array in arrays])
    limit = float(np.nanpercentile(values, percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.nanmax(values))
    return limit if limit > 0 else 1.0


def robust_speed_limit(*arrays, percentile):
    """Return a robust upper limit for speed histograms."""
    values = np.concatenate([flatten_speeds(array) for array in arrays])
    limit = float(np.nanpercentile(values, percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.nanmax(values))
    return limit if limit > 0 else 1.0


def describe(values):
    """Return compact distribution summary statistics."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def plot_histograms(items, output_path, bins, component_percentile, speed_percentile):
    """Plot predicted/reference velocity histograms for all inputs."""
    row_count = len(items)
    fig, axes = plt.subplots(row_count, 4, figsize=(18, max(3.2, 3.0 * row_count)), squeeze=False)

    column_titles = [
        "component v, first window",
        "component v, last window",
        "|v|, first window",
        "|v|, last window",
    ]
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title)

    for row, item in enumerate(items):
        label = item["label"]
        pred_begin = item["pred_begin"]
        pred_end = item["pred_end"]
        ref_begin = item["ref_begin"]
        ref_end = item["ref_end"]

        comp_limit = robust_component_limit(
            pred_begin,
            pred_end,
            ref_begin,
            ref_end,
            percentile=component_percentile,
        )
        speed_limit = robust_speed_limit(
            pred_begin,
            pred_end,
            ref_begin,
            ref_end,
            percentile=speed_percentile,
        )

        component_sets = [
            (flatten_components(pred_begin), flatten_components(ref_begin), (-comp_limit, comp_limit)),
            (flatten_components(pred_end), flatten_components(ref_end), (-comp_limit, comp_limit)),
        ]
        speed_sets = [
            (flatten_speeds(pred_begin), flatten_speeds(ref_begin), (0.0, speed_limit)),
            (flatten_speeds(pred_end), flatten_speeds(ref_end), (0.0, speed_limit)),
        ]

        for column, (pred, ref, hist_range) in enumerate(component_sets + speed_sets):
            ax = axes[row, column]
            ax.hist(ref, bins=bins, range=hist_range, density=True, alpha=0.45, label="reference", color="tab:blue")
            ax.hist(pred, bins=bins, range=hist_range, density=True, alpha=0.45, label="predicted", color="tab:orange")
            ax.grid(alpha=0.2)
            if row == row_count - 1:
                ax.set_xlabel("velocity")
            if column == 0:
                ax.set_ylabel(label)
            if row == 0 and column == 0:
                ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    return fig


def main():
    """Load inputs, plot histograms, and print summary statistics."""
    args = parse_args()
    labels = args.label
    if labels is None:
        labels = [Path(path).stem for path in args.input_path]
    if len(labels) != len(args.input_path):
        raise ValueError("--label must be repeated exactly once per --input-path")

    items = []
    print(
        "\t".join(
            [
                "label",
                "window",
                "source",
                "speed_mean",
                "speed_std",
                "speed_p95",
                "speed_p99",
                "speed_max",
                "component_std",
            ]
        )
    )
    for input_path, label in zip(args.input_path, labels):
        data = np.load(input_path)
        pred = load_displacements(data, predicted=True)
        ref = load_displacements(data, predicted=False)
        pred_begin, pred_end = velocity_windows(pred, args.dt, args.window_frames)
        ref_begin, ref_end = velocity_windows(ref, args.dt, args.window_frames)
        items.append(
            {
                "label": label,
                "pred_begin": pred_begin,
                "pred_end": pred_end,
                "ref_begin": ref_begin,
                "ref_end": ref_end,
            }
        )

        for window, source, values in [
            ("begin", "predicted", pred_begin),
            ("begin", "reference", ref_begin),
            ("end", "predicted", pred_end),
            ("end", "reference", ref_end),
        ]:
            speed_stats = describe(flatten_speeds(values))
            component_std = float(np.std(flatten_components(values)))
            print(
                "\t".join(
                    [
                        label,
                        window,
                        source,
                        f"{speed_stats['mean']:.8g}",
                        f"{speed_stats['std']:.8g}",
                        f"{speed_stats['p95']:.8g}",
                        f"{speed_stats['p99']:.8g}",
                        f"{speed_stats['max']:.8g}",
                        f"{component_std:.8g}",
                    ]
                )
            )

    fig = plot_histograms(
        items=items,
        output_path=args.output_path,
        bins=args.bins,
        component_percentile=args.percentile,
        speed_percentile=args.speed_percentile,
    )
    print(f"Saved {args.output_path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
