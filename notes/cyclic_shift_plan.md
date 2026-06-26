# Plan: cyclic shift

This note records the postponed "cyclic shift" scaling experiment.

## Idea

Use the existing flat RNN in no-overlap mode, but change the no-overlap tiling
offset at every autoregressive step. With periodic boundaries, every step still
covers the whole crystal exactly once, but block boundaries move through the
crystal instead of staying fixed.

## Motivation

Fixed no-overlap inference performed better than overlap stitching in the
previous experiments, but it keeps block boundaries at the same physical cells
for the entire rollout. A cyclic offset schedule may spread boundary artifacts
over time and let neighboring regions mix through subsequent steps.

## Candidate Schedules

1. Fixed no-overlap baseline:
   `(0, 0, 0)` at every step.

2. X-only shift:
   `(t mod 3, 0, 0)` for a `3x3x3` training block.

3. Diagonal shift:
   `(t mod 3, t mod 3, t mod 3)`.

4. Full 27-offset cycle:
   iterate over all offsets
   `(ox, oy, oz)` with `ox, oy, oz in {0, 1, 2}`.

The full 27-offset cycle is the most symmetric option. Over time, each physical
cell appears at many local block positions instead of always living near the
same no-overlap block boundary.

## Evaluation

Run the same comparison as before on `data1055.npz`, crop `9x3x3`:

- fixed no-overlap baseline;
- x-only cyclic shift;
- diagonal cyclic shift;
- full 27-offset cycle.

Compare:

- `S(q,w)` correlation with `NCELLS=9`, `KCOUNT=9`;
- q-band metrics from `analyze_sqw_q_bands.py`;
- visual `S(q,w)` maps.

## Expected Signal

If this works, it should reduce static block-boundary artifacts without needing
overlap averaging or Fourier low-q correction. If it fails, fixed no-overlap may
remain the best purely blockwise use of the flat RNN.

## 2026-05-10 Control Run

Implemented `infer_cyclic_shift.py` with four schedules:

- `fixed`;
- `x`;
- `diagonal`;
- `cycle`.

The `fixed` schedule was checked against the previous no-overlap/tile code on a
short rollout. The maximum absolute difference was `1.0430813e-07`, so the new
script reproduces the baseline.

Full comparison on `data1055.npz`, crop `9x3x3`, `count_steps=2998`,
`NCELLS=9`, `KCOUNT=9`:

| schedule | S(q,w) corr | low-q corr | low-q power |
| --- | ---: | ---: | ---: |
| fixed no-overlap | `0.589525` | `0.452248` | `1.019699` |
| x shift | `-0.065050` | `-0.177538` | `0.527445` |
| diagonal shift | `-0.092457` | `0.027668` | `0.435987` |
| full 27-offset cycle | `0.232268` | `0.117128` | `2.772502` |

Conclusion: cyclically moving no-overlap boundaries strongly degrades the
trajectory. The likely issue is that the flat RNN is not translation-equivariant
inside the `3x3x3` block: shifting the block changes the local position at
which each physical cell is predicted, and those position-specific biases are
mixed into the dynamics. Fixed no-overlap remains better for this flat RNN.
