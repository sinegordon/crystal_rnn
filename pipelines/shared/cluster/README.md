# Shared Cluster Helpers

- `submit_edge_search.py`: local submitter for pair-force and pair-energy
  Slurm-array searches through `cluster/run_edge_rnn_search.sh`.
- `check_edge_search.py`: local checker for pair-force and pair-energy search
  progress and current `top10.txt`.
- `run_cluster_ase_1055.py`: generic submit/wait/fetch wrapper for ASE 1055
  inference.

Paradigm-specific folders call these helpers with fixed defaults.
