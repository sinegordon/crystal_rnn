"""Compare predicted/reference S(q,w) quality in separate q bands."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from base_classes import get_sqw
from plot_sqw_comparison import (
    build_k_vectors,
    correlation,
    load_positions,
)


DT = 0.02
STEP = 10
LATTICE_PARAMETER = 3.615
NCELLS = 9
KCOUNT = 9
BANDS = ("low", "mid", "high")


def parse_args():
    """Parse command-line options for q-band S(q,w) diagnostics."""
    parser = argparse.ArgumentParser(
        description="Measure S(q,w) agreement separately for low/mid/high q bands."
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs=2,
        metavar=("LABEL", "NPZ_PATH"),
        required=True,
        help="Mode label and inference output .npz path. Can be passed multiple times.",
    )
    parser.add_argument("--output-prefix", default="inference_outputs/sqw_q_bands")
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--lattice-parameter", type=float, default=LATTICE_PARAMETER)
    parser.add_argument("--ncells", type=int, default=NCELLS)
    parser.add_argument("--kcount", type=int, default=KCOUNT)
    parser.add_argument(
        "--band-fractions",
        type=float,
        nargs=2,
        default=(1 / 3, 2 / 3),
        metavar=("LOW_END", "MID_END"),
        help="Quantile-like split points over sorted q columns.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def relative_error(predicted, reference):
    """Return normalized L2 error for one S(q,w) slice."""
    denominator = np.linalg.norm(reference.reshape(-1))
    if denominator == 0:
        return np.nan
    return float(np.linalg.norm((predicted - reference).reshape(-1)) / denominator)


def power_ratio(predicted, reference):
    """Return total predicted S(q,w) power divided by reference power."""
    denominator = float(np.nansum(reference))
    if denominator == 0:
        return np.nan
    return float(np.nansum(predicted) / denominator)


def build_band_slices(q_values, split_fractions):
    """Split sorted q columns into low/mid/high bands."""
    order = np.argsort(q_values)
    count = len(order)
    low_end = max(1, int(round(count * split_fractions[0])))
    mid_end = max(low_end + 1, int(round(count * split_fractions[1])))
    mid_end = min(mid_end, count - 1)
    return {
        "low": order[:low_end],
        "mid": order[low_end:mid_end],
        "high": order[mid_end:],
    }


def calculate_sqw(path, kmas, dt, step):
    """Load one inference output and compute predicted/reference S(q,w)."""
    data = np.load(path)
    predicted_positions = load_positions(data, "predicted_positions", "predicted_displacements")
    reference_positions = load_positions(data, "reference_positions_output", "reference_displacements")
    xi_pred, yi_pred, predicted_sqw = get_sqw(predicted_positions, dt=dt, step=step, kmas=kmas)
    xi_ref, yi_ref, reference_sqw = get_sqw(reference_positions, dt=dt, step=step, kmas=kmas)
    if not np.allclose(xi_pred, xi_ref) or not np.allclose(yi_pred, yi_ref):
        raise ValueError(f"S(q,w) grids do not match for {path}")
    return xi_pred, yi_pred, predicted_sqw, reference_sqw


def collect_metrics(label, predicted_sqw, reference_sqw, q_values, band_columns):
    """Calculate all-band and per-band metrics for one mode."""
    rows = []
    band_map = {"all": np.arange(len(q_values)), **band_columns}
    for band, columns in band_map.items():
        predicted_band = predicted_sqw[:, columns]
        reference_band = reference_sqw[:, columns]
        rows.append(
            {
                "mode": label,
                "band": band,
                "q_min": float(np.min(q_values[columns])),
                "q_max": float(np.max(q_values[columns])),
                "corr": correlation(predicted_band, reference_band),
                "rel_l2": relative_error(predicted_band, reference_band),
                "power_ratio": power_ratio(predicted_band, reference_band),
                "mean_abs_error": float(np.nanmean(np.abs(predicted_band - reference_band))),
                "signed_error": float(np.nanmean(predicted_band - reference_band)),
            }
        )
    return rows


def write_table(rows, output_path):
    """Write metric rows as a TSV file."""
    fields = (
        "mode",
        "band",
        "q_min",
        "q_max",
        "corr",
        "rel_l2",
        "power_ratio",
        "mean_abs_error",
        "signed_error",
    )
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append(
            "\t".join(
                str(row[field]) if isinstance(row[field], str) else f"{row[field]:.8g}"
                for field in fields
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(rows, q_profiles, output_path):
    """Plot q-band metrics and q-resolved average absolute error."""
    modes = list(dict.fromkeys(row["mode"] for row in rows))
    bands = ["low", "mid", "high"]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(modes)))
    color_by_mode = dict(zip(modes, colors))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = np.arange(len(bands))
    width = 0.8 / max(1, len(modes))

    for mode_index, mode in enumerate(modes):
        offset = (mode_index - (len(modes) - 1) / 2) * width
        band_rows = {
            row["band"]: row
            for row in rows
            if row["mode"] == mode and row["band"] in bands
        }
        axes[0, 0].bar(
            x + offset,
            [band_rows[band]["corr"] for band in bands],
            width,
            label=mode,
            color=color_by_mode[mode],
        )
        axes[0, 1].bar(
            x + offset,
            [band_rows[band]["rel_l2"] for band in bands],
            width,
            color=color_by_mode[mode],
        )
        axes[1, 0].bar(
            x + offset,
            [band_rows[band]["power_ratio"] for band in bands],
            width,
            color=color_by_mode[mode],
        )

    axes[0, 0].set_title("Correlation by q band")
    axes[0, 0].set_ylabel("Pearson corr")
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Relative L2 error by q band")
    axes[0, 1].set_ylabel("||pred-ref|| / ||ref||")
    axes[1, 0].set_title("Power ratio by q band")
    axes[1, 0].set_ylabel("sum(pred) / sum(ref)")
    axes[1, 0].axhline(1, color="black", linewidth=0.8, linestyle="--")

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(x)
        axis.set_xticklabels(bands)

    for mode, (q_values, mean_abs_error, signed_error) in q_profiles.items():
        axes[1, 1].plot(
            q_values,
            mean_abs_error,
            marker="o",
            label=f"{mode} abs",
            color=color_by_mode[mode],
        )
        axes[1, 1].plot(
            q_values,
            np.abs(signed_error),
            marker="x",
            linestyle="--",
            color=color_by_mode[mode],
            alpha=0.65,
        )
    axes[1, 1].set_title("Frequency-averaged error vs |q|")
    axes[1, 1].set_xlabel("|q|")
    axes[1, 1].set_ylabel("mean error over energy")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=min(3, len(modes)))
    fig.savefig(output_path, dpi=200)
    return fig


def main():
    """Run the q-band S(q,w) diagnostic."""
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    kmas = build_k_vectors(args.ncells, args.lattice_parameter, args.kcount)
    q_values = np.linalg.norm(kmas, axis=1)
    band_columns = build_band_slices(q_values, args.band_fractions)

    rows = []
    q_profiles = {}
    for label, path in args.case:
        _, _, predicted_sqw, reference_sqw = calculate_sqw(Path(path), kmas, args.dt, args.step)
        rows.extend(collect_metrics(label, predicted_sqw, reference_sqw, q_values, band_columns))
        q_profiles[label] = (
            q_values,
            np.nanmean(np.abs(predicted_sqw - reference_sqw), axis=0),
            np.nanmean(predicted_sqw - reference_sqw, axis=0),
        )

    table_path = output_prefix.with_suffix(".tsv")
    image_path = output_prefix.with_suffix(".png")
    write_table(rows, table_path)
    fig = plot_summary(rows, q_profiles, image_path)

    print(f"Saved {table_path}")
    print(f"Saved {image_path}")
    print(table_path.read_text(encoding="utf-8"))

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
