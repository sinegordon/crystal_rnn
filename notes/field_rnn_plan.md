# Plan: recurrent-convolutional field RNN

## Idea

Keep recurrent temporal processing, but remove dense position-specific block
channels:

- encode each crystal frame with periodic 3D convolutions;
- apply the same temporal RNN/GRU/LSTM to every unit-cell location;
- decode the resulting hidden field with periodic 3D convolutions.

The model predicts a full crystal field directly and can be applied to a larger
rectangular crystal without block stitching.

## Implementation

- `base_classes/field_rnn_predictor.py` adds `CrystalFieldRNNNet`.
- `find_field_rnn_models.py` trains/searches this architecture and scores it
  by the same `S(q,w)` workflow used by the other scripts.

## 2026-05-11 First Check

Smoke tests passed:

- `py_compile`;
- synthetic train + rollout;
- tiny `data333.npz` S(q,w) smoke.

First non-toy candidate:

- model: `GRU`;
- encoder channels: `32`;
- RNN hidden size: `64`;
- RNN layers: `1`;
- conv layers: `1`;
- kernel size: `3`;
- target mode: `absolute_delta`;
- delta loss weight: `3`;
- train window: `10000` frames;
- data fraction: `0.2`;
- epochs: `30`;
- batch size: `128`.

Results:

| dataset | geometry | metric | value |
| --- | --- | --- | ---: |
| `data333.npz` | `3x3x3` | `S(q,w)` norm | `2.9565458639699322` |
| `data1055.npz` | `10x5x5` full field | `S(q,w)` corr, `NCELLS=10`, `KCOUNT=10` | `0.285757` |

For `data1055.npz`, q-band diagnostics show low-q is still weak:

| band | corr | rel L2 | power ratio |
| --- | ---: | ---: | ---: |
| low | `0.04133387` | `2.1888006` | `3.8011351` |
| mid | `0.28033399` | `1.8690067` | `3.774028` |
| high | `0.39400892` | `1.783796` | `3.3866015` |

Conclusion: the first field-RNN candidate is stable enough to run directly on
the larger crystal, but it overpowers the spectrum and still fails mainly in
the long-wavelength region. It is not competitive yet, but the architecture is
now implemented and ready for parameter/loss experiments.
