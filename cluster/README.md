# Cluster Runs

This folder contains Slurm helpers for running the acceleration-normalized
`CrystalFieldRNNNet` search on a cluster.

## Search 20 Models

From the repository root on the cluster:

```bash
sbatch cluster/run_field_rnn_accnorm_array.slurm
```

The default job array is `0-19`, so it trains 20 independent one-model jobs.
The script submits to the `gold-batch` partition, where the GPU node is expected
to be `node54`.
Each task uses:

```bash
python find_field_rnn_models.py 1 RNN \
  --data-path data333.npz \
  --target-mode acceleration \
  --acceleration-normalization channel \
  --bidirectional \
  --epochs 30 \
  --batch-size 128 \
  --data-len 0.2 \
  --device auto
```

`--device auto` uses CUDA when PyTorch sees it, otherwise CPU.

## Common Overrides

Use environment variables before `sbatch`:

```bash
CONDA_ENV=torch \
DATA_PATH=data333.npz \
EPOCHS=50 \
BATCH_SIZE=128 \
sbatch --array=0-9 cluster/run_field_rnn_accnorm_array.slurm
```

If the cluster needs modules or an explicit Conda init script:

```bash
MODULE_LOAD="cuda/12.1 anaconda3" \
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh" \
CONDA_ENV=torch \
sbatch cluster/run_field_rnn_accnorm_array.slurm
```

To force CPU:

```bash
DEVICE=cpu sbatch --array=0-9 cluster/run_field_rnn_accnorm_array.slurm
```

To train only on the central unit cell of each `3x3x3` block:

```bash
LOSS_REGION=center_cell sbatch --array=0-9 cluster/run_field_rnn_accnorm_array.slurm
```

To penalize models that systematically overestimate acceleration or next-step
velocity amplitudes:

```bash
ACCELERATION_RMS_LOSS_WEIGHT=0.1 \
VELOCITY_RMS_LOSS_WEIGHT=0.1 \
sbatch --array=0-9 cluster/run_field_rnn_accnorm_array.slurm
```

To pass extra CLI arguments directly to `find_field_rnn_models.py`, append them
after the Slurm script name:

```bash
sbatch --array=0-9 cluster/run_field_rnn_accnorm_array.slurm --rnn-hidden-size 32
```

## Outputs

By default outputs are written to:

```text
models333_field_rnn_accnorm_cluster/
inference_outputs/field_rnn_accnorm_cluster/
logs/
```

After the array finishes, collect the per-task metric files:

```bash
python cluster/collect_field_rnn_metrics.py
```

This writes:

```text
inference_outputs/field_rnn_accnorm_cluster/summary.tsv
```
