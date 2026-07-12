# Pair-Energy Pipeline

The article-only, one-frame implementation is available in
[`mlp/README.md`](mlp/README.md). It provides an independent model class, data
format, search, stateless ASE calculator, postprocessing, and cluster launchers
without RNN compatibility arguments.

This branch exposes one stable model pipeline: conservative `pair_energy`.

Public entrypoints:

- `pair_energy/data/prepare_force_data.py`
- `pair_energy/search/find_models.py`
- `pair_energy/cluster/submit_search.py`
- `pair_energy/cluster/check_search.py`
- `pair_energy/cluster/submit_inference_1055.py`
- `pair_energy/cluster/fetch_inference_1055.py`
- `pair_energy/postprocess/plot_etot_heat_capacity.py`
- `pair_energy/postprocess/plot_etot_cumulative.py`

Shared implementation wrappers live in `shared/`.  They are not separate model
pipelines; they provide cluster configuration, ASE launch wrappers, data
helpers, and postprocessing scripts used by `pair_energy`.

## Cluster Configuration

Pipeline cluster launchers read SSH, remote path, Slurm, GPU-node, and conda
defaults from:

```text
pipelines/shared/cluster/cluster_config.json
```

Override it with `--cluster-config /path/to/config.json` or with the
`CRYSTAL_RNN_CLUSTER_CONFIG` environment variable.  Explicit command-line
arguments such as `--host`, `--nodelist`, `--partition`, and `--remote-workdir`
still override the config for a single run.

## Current Recommended Path

- train on force-enabled `3x3x3` Cu data;
- select by `S(q,w)` rollout score on `333`;
- run ASE 1055 NVT with Bussi thermostat;
- use direct model `E_tot = E_pot_model + E_kin` as the primary energy
  diagnostic;
- keep path-energy as a force/work consistency diagnostic.

The clean ASE inference preset is:

- `T = 300 K`;
- `dt = 0.002 ps`;
- Bussi thermostat coupling `taut = 200 fs`;
- `q_zero_mode = initial`, meaning initial COM-velocity removal only;
- no history damping, adaptive eta, curl correction, or power-bias correction.

See [pair_energy/README.md](pair_energy/README.md) for the full command set.
