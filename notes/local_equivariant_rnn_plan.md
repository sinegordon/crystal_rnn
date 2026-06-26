# Plan: local translation-equivariant RNN

## Idea

Replace the block-to-block RNN map with one shared local operator:

- input: a history of local crystal patches around one unit cell;
- output: the next displacement of only the central unit cell;
- rollout: apply the same operator to every unit cell at every step.

This removes position-specific output channels inside a `3x3x3` block.  A cell
predicted at the left edge of one tiling and a cell predicted at the center of
another tiling now use the same network weights, provided their local patches
are the same up to translation.

## First implementation

- `CrystalLocalRNNNet` stores `patch_shape`, `unit_cell_atoms`,
  `flatten_order`, and `target_mode`.
- The underlying `RNNNet` now supports `out_features`, so the local model can
  accept a full patch vector while returning only one unit-cell vector.
- `make_local_patch_samples` samples random `(time, center_cell)` examples from
  a crystal trajectory.
- `find_local_rnn_models.py` trains local models and can evaluate them by the
  same `S(q,w)` norm used by `find_models.py`.

## Expected signal

If the cyclic-shift failure was caused by the old flat RNN not being
translation-equivariant inside the block, this model should be less sensitive
to where artificial block boundaries fall.  It may still fail if the missing
long-wavelength dynamics cannot be reconstructed from local displacement
history alone.

## 2026-05-10 First Check

Implemented the local model and a first search script:

- `base_classes/local_rnn_predictor.py`;
- `find_local_rnn_models.py`.

Smoke tests:

- synthetic crystal train + rollout passed;
- real `data333.npz` CLI smoke passed.
- saved-model inference through `infer_model.py` passed.

First non-toy candidate on `data333.npz`:

- model: `RNN`, hidden size `100`, layers `3`;
- patch shape: `3x3x3`;
- target mode: `absolute_delta`;
- train samples: `20000`;
- epochs: `20`;
- batch size: `500`;
- rollout: `count_steps=2000`, `count_run=1`.

Metrics:

| metric | value |
| --- | ---: |
| `S(q,w)` norm | `2.893173521824467` |
| one-step rel L2 | `0.08990946412086487` |
| delta corr | `-0.33873489516987537` |
| delta rel L2 | `1.784698486328125` |
| delta std ratio | `1.1698021526529367` |

The first result is not yet better than the best old flat RNNs, but it is in
the same broad range after only one candidate and noticeably better than the
short smoke run.  The negative delta correlation is the main warning sign:
absolute positions are learned reasonably, while one-step local increments are
not yet aligned well.
