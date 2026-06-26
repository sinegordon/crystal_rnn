#!/usr/bin/env python3
"""Collect Slurm-array field-RNN search metrics into one sorted TSV."""

import argparse
import csv
from pathlib import Path


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        default="inference_outputs/field_rnn_accnorm_cluster",
        help="Directory containing metrics_*.tsv files.",
    )
    parser.add_argument(
        "--output-path",
        default="inference_outputs/field_rnn_accnorm_cluster/summary.tsv",
        help="Path for the merged, score-sorted TSV.",
    )
    return parser.parse_args()


def read_rows(metrics_dir):
    """Read all per-task metrics rows."""
    rows = []
    for path in sorted(Path(metrics_dir).glob("metrics_*.tsv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                row["metrics_file"] = str(path)
                rows.append(row)
    return rows


def row_score(row):
    """Return the preferred model-selection score for sorting."""
    if "selection_score" in row and row["selection_score"]:
        return float(row["selection_score"])
    return float(row["sqw_norm"])


def write_rows(rows, output_path):
    """Write rows sorted by the available model-selection score."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=row_score)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    fields = ["metrics_file", *[field for field in fields if field != "metrics_file"]]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return rows


def main():
    """Collect and print the best models."""
    args = parse_args()
    rows = read_rows(args.metrics_dir)
    if not rows:
        raise SystemExit(f"No metrics_*.tsv files found in {args.metrics_dir}")
    rows = write_rows(rows, args.output_path)
    print(f"Saved {args.output_path}")
    print("Best models:")
    for row in rows[:10]:
        velocity_score = row.get("velocity_score", "")
        acceleration_ratio = row.get("acceleration_rms_ratio", "")
        if velocity_score:
            acceleration_text = (
                f"acc_rms={float(acceleration_ratio):.6g}\t"
                if acceleration_ratio
                else ""
            )
            scale_text = ""
            if row.get("velocity_pred_rms") and row.get("velocity_ref_rms"):
                scale_text += (
                    f"vel_model={float(row['velocity_pred_rms']):.6g}\t"
                    f"vel_ref={float(row['velocity_ref_rms']):.6g}\t"
                )
            if row.get("acceleration_pred_rms") and row.get("acceleration_ref_rms"):
                scale_text += (
                    f"acc_model={float(row['acceleration_pred_rms']):.6g}\t"
                    f"acc_ref={float(row['acceleration_ref_rms']):.6g}\t"
                )
            print(
                f"selection={row_score(row):.6g}\t"
                f"sqw={float(row['sqw_norm']):.6g}\t"
                f"velocity={float(velocity_score):.6g}\t"
                f"{acceleration_text}"
                f"{scale_text}"
                f"{row['model_path']}"
            )
        else:
            print(f"{float(row['sqw_norm']):.6g}\t{row['model_path']}")


if __name__ == "__main__":
    main()
