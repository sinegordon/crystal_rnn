# Standalone pair-energy MLP pipeline

This directory contains the article MLP interface without RNN compatibility
arguments or stateful inference code.

## Model contract

`CrystalPairEnergyMLPNet` receives one current displacement field. For every
atom in the central conventional FCC cell, it builds 18 first- and second-shell
pair features:

```text
[R_ref_ij / a0, (u_j - u_i) / a0] -> 256 -> ELU -> 256 -> ELU -> scalar U_ij
```

The scalar pair energy is made exchange-even, and pair forces are obtained by
differentiation. The same contribution is scattered with opposite signs to the
two atoms. The default `size=256` model has 67,841 trainable parameters.

The neural model uses exactly one physical frame. Two initial frames are
required only by the Verlet/ASE driver to reconstruct the initial velocity.
There is no hidden trajectory history in the ASE calculator.

## Files

- `data/prepare_data.py`: LAMMPS dump to one-frame force-training NPZ.
- `data/convert_legacy_npz.py`: validated conversion of a force-enabled legacy NPZ.
- `search/find_models.py`: local training, S(q,w) ranking, plots, and checkpoints.
- `ase/calculator.py`: stateless ASE calculator.
- `ase/run_nvt.py`: two-frame initialization and Bussi NVT inference.
- `postprocess/run_all.py`: all standard article diagnostics.
- `cluster/submit_search.py`: local launcher for a cluster Slurm search array.
- `cluster/check_search.py`: queue status and current top 10.
- `cluster/submit_inference.py`: local launcher for ASE inference and postprocessing.
- `cluster/fetch_inference.py`: status, logs, and result download.
- `convert_legacy_checkpoint.py`: exact conversion of the previous MLP1 checkpoint.

## Prepare data

```bash
python pipelines/mlp/data/prepare_data.py \
  /path/to/Cu333/dump.lammpstrj data333_mlp_force.npz \
  --crystal-shape 3 3 3 \
  --train-supercell-shape 3 3 3 \
  --dt-ps 0.002
```

The output stores `input_blocks` with shape
`(samples, 3, 3, 3, atoms_per_cell, 3)`. It deliberately has no temporal axis.
The search script can also read a legacy NPZ and uses the latest frame of each
legacy `X_blocks` sample.

If the force-enabled legacy NPZ is already available, create a compact explicit
MLP dataset without re-reading the LAMMPS dump:

```bash
python pipelines/mlp/data/convert_legacy_npz.py \
  data333_force_L1.npz data/data333_mlp_force.npz
```

## Search locally

```bash
conda run -n torch python pipelines/mlp/search/find_models.py 10 \
  --data-path data333_mlp_force.npz \
  --models-dir models333_mlp \
  --size 256 \
  --delta-frames 30000 \
  --epochs 50 \
  --count-steps 2000 \
  --count-run 3
```

Each checkpoint contains a format version, geometry, model configuration,
normalization values, and a `state_dict`; it does not pickle the legacy RNN
class.

## Convert a previous MLP1 model

```bash
python pipelines/mlp/convert_legacy_checkpoint.py \
  legacy_model.pth standalone_model.pth
```

The converter checks the architecture, maps all three affine layers, and
verifies predicted accelerations numerically before saving.

## Run ASE locally

```bash
python pipelines/mlp/ase/run_nvt.py \
  --model-path standalone_model.pth \
  --data-path data1055.npz \
  --output-npz outputs/mlp_1055/trajectory.npz \
  --steps 50000 \
  --initial-frames 10 11 \
  --temperature-k 300 \
  --taut-fs 200 \
  --dt-ps 0.002

python pipelines/mlp/postprocess/run_all.py \
  --ase-path outputs/mlp_1055/trajectory.npz \
  --data-path data1055.npz \
  --output-dir outputs/mlp_1055/postprocess \
  --ncells 10 --kcount 10
```

Initial COM velocity is removed once by default. No COM projection or internal
force correction is applied during subsequent steps.

## Run on the cluster

Cluster defaults come from `pipelines/shared/cluster/cluster_config.json`.
Copy `cluster/cluster_config.example.json` outside the repository and pass it
with `--cluster-config` to override the host, key, work directory, node, or
Conda environment.

```bash
python pipelines/mlp/cluster/submit_search.py \
  --cluster-config ~/.config/crystal_rnn/cluster.json \
  --model-count 10 \
  --data-path data333_mlp_force.npz \
  --delta-frames 30000

python pipelines/mlp/cluster/check_search.py logs/<label>_search_state.json
```

```bash
python pipelines/mlp/cluster/submit_inference.py \
  --cluster-config ~/.config/crystal_rnn/cluster.json \
  --model-path models333_mlp/best.pth \
  --data-path data1055.npz \
  --steps 50000 \
  --initial-frames 10 11 \
  --temperature-k 300 \
  --taut-fs 200

python pipelines/mlp/cluster/fetch_inference.py \
  logs/<label>_inference_state.json
```

Code synchronization excludes `.npz`, `.pth`, models, logs, and output
directories. Datasets and selected checkpoints must therefore already exist
under `remote_workdir`; they are never overwritten by a launcher.
