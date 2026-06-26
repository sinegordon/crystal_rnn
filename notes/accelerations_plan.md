# Plan: accelerations

This note records the postponed "accelerations" research path.

## Question

Can the existing flat RNN become a usable acceleration operator if it is trained
harder and selected by acceleration quality instead of only by long-rollout
`S(q,w)`?

## Baseline Observation

The previous acceleration-oriented experiments did not work well:

- `target_mode=acceleration` produced poor `S(q,w)` scores.
- `target_mode=verlet` also did not materially improve the result.
- Adding an auxiliary acceleration loss to `absolute_delta` barely changed the
  one-step acceleration scale.
- One-step diagnostics showed `pred_acceleration.std()` was hundreds of times
  larger than `reference_acceleration.std()` for the tested models.

## Proposed Control Experiment

1. Train `target_mode=acceleration` on much more data than the current default
   randomized slice.
2. Use a larger training window or multiple windows instead of one small
   `DELTA` segment.
3. Increase epochs and possibly reduce learning rate.
4. Keep a validation split for one-step acceleration diagnostics.
5. Select candidate models by acceleration metrics before long `S(q,w)` rollout:
   correlation, relative L2 error, and predicted/reference standard-deviation
   ratio.
6. Compare first on `data333.npz`; only then test transfer to `data1055.npz`.

## Decision Criterion

If `pred_acceleration.std() / reference_acceleration.std()` remains orders of
magnitude too large on `data333`, the issue is likely architectural rather than
just insufficient training data.

## 2026-05-10 Control Run

Implemented `find_acceleration_models.py` and trained one `RNN` acceleration
model on almost all `data333.npz`:

- train samples: `97997`;
- validation samples: `2000`;
- epochs: `10`;
- batch size: `500`;
- learning rate: `0.001`.

Acceleration validation compared to the old `absolute_delta` model on the same
validation windows:

| model | acceleration corr | rel L2 | std ratio |
| --- | ---: | ---: | ---: |
| old `absolute_delta` | `0.589588` | `298.210977` | `298.799438` |
| new `acceleration` | `0.146042` | `2.341239` | `2.267910` |

The larger training run fixed the acceleration scale by roughly two orders of
magnitude, but acceleration correlation became much worse.

Long rollout on `data333.npz`, `count_steps=2998`, `NCELLS=3`, `KCOUNT=3`:

| model | S(q,w) corr |
| --- | ---: |
| old `absolute_delta` | `0.581208` |
| new `acceleration` | `-0.101013` |

Conclusion: more data helps the acceleration amplitude dramatically, but the
flat RNN acceleration operator still does not produce stable/useful dynamics in
rollout. The next acceleration-path attempt would need better acceleration
correlation or a different architecture/regularization, not just more samples.
