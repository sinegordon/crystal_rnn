# Crystal pair-energy MLP

This branch isolates the one-frame conservative MLP interface used by the
article baseline. The primary implementation and all reproducible entrypoints
are documented in [pipelines/mlp/README.md](pipelines/mlp/README.md).

## Physical model

The fixed FCC lattice frame defines 18 first- and second-shell neighbors for
each atom in the central conventional cell. The model consumes

```text
[R_ref_ij / a0, (u_j - u_i) / a0]
```

from one current frame and predicts an exchange-even scalar pair energy.
Forces are its gradient with respect to the dynamic pair-vector channels.
Unique-pair scatter gives equal and opposite atomic forces by construction.

The article configuration is an ELU MLP with widths `6 -> 256 -> 256 -> 1`
and 67,841 trainable parameters. It contains no recurrent layer and no hidden
trajectory state.

## Public interface

- `base_classes.CrystalPairEnergyMLPNet`: training, force/energy prediction,
  periodic full-crystal inference, two-frame Verlet rollout, and portable
  checkpoints.
- `pipelines/mlp/data/prepare_data.py`: one-frame force dataset preparation.
- `pipelines/mlp/search/find_models.py`: candidate search and S(q,w) ranking.
- `pipelines/mlp/ase/run_nvt.py`: stateless ASE/Bussi NVT dynamics.
- `pipelines/mlp/postprocess/run_all.py`: complete article diagnostics.
- `pipelines/mlp/cluster/`: local launch, status, and fetch tools for Slurm.

Two physical frames are accepted by the dynamics driver only to reconstruct
the initial velocity. The neural force operator itself always receives the
current frame alone.

## Quick check

```bash
conda run -n torch python -m unittest tests.test_standalone_pair_energy_mlp
```

Legacy one-frame MLP1 checkpoints can be converted exactly with
`pipelines/mlp/convert_legacy_checkpoint.py`; the converter verifies numerical
force agreement before writing the standalone checkpoint.
