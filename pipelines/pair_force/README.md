# Pair-Force Pipeline

Pair-force models predict antisymmetric pair contributions and sum/scatter them
to obtain accelerations.

Main entrypoints:

- `data/prepare_force_data.py`: prepare force-enabled crystal `.npz` data.
- `search/find_models.py`: pair-force model search.
- `inference/edge_rnn.py`: direct pair/edge rollout helper.
- `cluster/submit_search.py`: submit pair-force search to the cluster.
- `cluster/check_search.py`: inspect pair-force search progress.
- `cluster/submit_inference_1055.py`: submit ASE 1055 inference.
- `cluster/fetch_inference_1055.py`: check/fetch ASE 1055 inference outputs.
