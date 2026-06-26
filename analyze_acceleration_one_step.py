"""Diagnose one-step model-derived acceleration fields from inference outputs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_sqw_comparison import correlation


def parse_args():
    """Parse command-line options for one-step acceleration diagnostics."""
    parser = argparse.ArgumentParser(
        description="Compare model-derived and reference discrete accelerations for one predicted step."
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs=2,
        metavar=("LABEL", "NPZ_PATH"),
        required=True,
        help="Mode label and inference output .npz path. Can be passed multiple times.",
    )
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--output-prefix", default="inference_outputs/acceleration_one_step")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def relative_error(predicted, reference):
    """Return normalized L2 error."""
    denominator = np.linalg.norm(reference.reshape(-1))
    if denominator == 0:
        return np.nan
    return float(np.linalg.norm((predicted - reference).reshape(-1)) / denominator)


def power_ratio(predicted, reference):
    """Return total squared-amplitude ratio."""
    predicted_power = float(np.sum(predicted**2))
    reference_power = float(np.sum(reference**2))
    if reference_power == 0:
        return np.nan
    return predicted_power / reference_power


def load_accelerations(path, step_index):
    """Load predicted/reference discrete accelerations from an inference output file."""
    data = np.load(path)
    required = ["init_displacements", "predicted_displacements", "reference_displacements"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{path} is missing arrays: {missing}")
    if step_index >= data["predicted_displacements"].shape[0]:
        raise ValueError(f"step_index={step_index} is outside predicted trajectory in {path}")

    if step_index == 0:
        previous = data["init_displacements"][-2]
        current = data["init_displacements"][-1]
    else:
        previous = data["predicted_displacements"][step_index - 2] if step_index > 1 else data["init_displacements"][-1]
        current = data["predicted_displacements"][step_index - 1]

    predicted_next = data["predicted_displacements"][step_index]
    reference_next = data["reference_displacements"][step_index]
    predicted_acceleration = predicted_next - 2 * current + previous
    reference_acceleration = reference_next - 2 * current + previous
    return predicted_acceleration.astype(np.float64), reference_acceleration.astype(np.float64)


def spatial_power_by_q(field):
    """Return q magnitudes and atom/coordinate-summed FFT power."""
    spectrum = np.fft.fftn(field, axes=(0, 1, 2))
    power = np.sum(np.abs(spectrum) ** 2, axis=(3, 4))
    q_axes = np.meshgrid(
        *[np.fft.fftfreq(size) for size in field.shape[:3]],
        indexing="ij",
    )
    q_abs = np.sqrt(sum(axis**2 for axis in q_axes))
    q_values = np.unique(q_abs)
    q_power = np.array([np.mean(power[np.isclose(q_abs, q_value)]) for q_value in q_values])
    return q_values, q_power


def collect_rows(label, predicted_acceleration, reference_acceleration):
    """Calculate scalar one-step acceleration metrics."""
    return {
        "mode": label,
        "corr": correlation(predicted_acceleration, reference_acceleration),
        "rel_l2": relative_error(predicted_acceleration, reference_acceleration),
        "power_ratio": power_ratio(predicted_acceleration, reference_acceleration),
        "pred_std": float(np.std(predicted_acceleration)),
        "ref_std": float(np.std(reference_acceleration)),
        "mean_bias": float(np.mean(predicted_acceleration - reference_acceleration)),
    }


def write_table(rows, output_path):
    """Write diagnostic rows as TSV."""
    fields = ("mode", "corr", "rel_l2", "power_ratio", "pred_std", "ref_std", "mean_bias")
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append(
            "\t".join(
                str(row[field]) if isinstance(row[field], str) else f"{row[field]:.8g}"
                for field in fields
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_power(power_profiles, output_path):
    """Plot predicted/reference spatial acceleration power by q magnitude."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, (q_pred, pred_power, q_ref, ref_power) in power_profiles.items():
        axes[0].plot(q_pred, pred_power / np.max(pred_power), marker="o", label=f"{label} pred")
        axes[1].plot(q_pred, pred_power / np.maximum(ref_power, 1e-30), marker="o", label=label)
    first_ref = next(iter(power_profiles.values()))
    axes[0].plot(first_ref[2], first_ref[3] / np.max(first_ref[3]), color="black", linewidth=2, label="reference")
    axes[0].set_title("Normalized acceleration spatial power")
    axes[0].set_xlabel("FFT |q| index")
    axes[0].set_ylabel("power / max(power)")
    axes[1].set_title("Predicted/reference power ratio")
    axes[1].set_xlabel("FFT |q| index")
    axes[1].set_ylabel("ratio")
    axes[1].axhline(1, color="black", linewidth=0.8, linestyle="--")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(output_path, dpi=200)
    return fig


def main():
    """Run one-step acceleration diagnostics."""
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    power_profiles = {}
    for label, path in args.case:
        predicted_acceleration, reference_acceleration = load_accelerations(Path(path), args.step_index)
        rows.append(collect_rows(label, predicted_acceleration, reference_acceleration))
        q_pred, pred_power = spatial_power_by_q(predicted_acceleration)
        q_ref, ref_power = spatial_power_by_q(reference_acceleration)
        if not np.allclose(q_pred, q_ref):
            raise ValueError(f"Spatial q grids do not match for {path}")
        power_profiles[label] = (q_pred, pred_power, q_ref, ref_power)

    table_path = output_prefix.with_suffix(".tsv")
    image_path = output_prefix.with_suffix(".png")
    write_table(rows, table_path)
    fig = plot_power(power_profiles, image_path)

    print(f"Saved {table_path}")
    print(f"Saved {image_path}")
    print(table_path.read_text(encoding="utf-8"))

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
