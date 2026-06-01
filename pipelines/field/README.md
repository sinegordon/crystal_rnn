# Field RNN Pipeline

Field models operate on crystal-shaped displacement fields and predict
accelerations or next displacements on the grid.

Main entrypoints:

- `data/prepare_crystal_data.py`: prepare crystal `.npz` data.
- `search/find_models.py`: run `find_field_rnn_models.py`.
- `inference/centered_acceleration.py`: centered local FieldRNN inference.
- `inference/centered_ensemble.py`: ensemble centered FieldRNN inference.
- `cluster/submit_search.py`: cluster model-search workflow.
