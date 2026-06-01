# Shared Cluster Helpers

- `cluster_config.json`: the default local cluster configuration used by
  pipeline submit/check/fetch scripts.
- `config.py`: shared JSON loader for cluster defaults.
- `submit_edge_search.py`: local submitter for pair-force and pair-energy
  Slurm-array searches through `cluster/run_edge_rnn_search.sh`.
- `check_edge_search.py`: local checker for pair-force and pair-energy search
  progress and current `top10.txt`.
- `run_cluster_ase_1055.py`: generic submit/wait/fetch wrapper for ASE 1055
  inference.

Paradigm-specific folders call these helpers with fixed defaults.

## Configuration

Cluster defaults are read from `cluster_config.json`:

```json
{
  "ssh": {
    "host": "sinegordon@cluster.vstu.ru",
    "port": "57322",
    "identity_file": "~/.ssh/id_ed25519_cluster_vstu"
  },
  "paths": {
    "remote_workdir": "~/crystal_rnn_accnorm"
  },
  "slurm": {
    "default_partition": "gold-batch",
    "gpu_nodelist": "node54.cluster",
    "train_partition": "gold-batch",
    "train_nodelist": "node54.cluster",
    "collect_partition": "gold-batch",
    "conda_env": "torch"
  }
}
```

All values can still be overridden from the command line, for example
`--host`, `--partition`, `--nodelist`, or `--remote-workdir`.  To use a
different config without editing the repository, pass `--cluster-config
/path/to/config.json` or set `CRYSTAL_RNN_CLUSTER_CONFIG=/path/to/config.json`.
